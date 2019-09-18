"""Pool with a map fuction.

About 10% faster on the slow main process than standard pool map.

Additional customizable task shuffle function is available that can be used
for worker balancing or specific task preparation.
For example, group tasks and feed map workers that groups as new tasks.

Arch:
Pool starts multiprocessor.
Multiprocessor starts dispatchers.
Dispatchers starts collectors/mergers and pools of workers.

Pool send functions and tasks to the multiprocessor.
Multiprocessor make batches, splits every batch by dispatchers tasks and feed them.
Dispatchers shuffler/prepare tasks and feed workers.
Workers invoke map fuction worker and feed collectors.
Collectors merge results if necessary and feed them to multiprocessor.
Multiprocessor return batch to the Pool.
"""

from __future__ import division

import collections
import datetime
import traceback
import marshal

from itertools import chain, cycle

from multiprocessing import (
    Pool as StandardPool, JoinableQueue, Pipe, Process,
    cpu_count, log_to_stderr, current_process
)


def log_info(who, text):
    """Helper logging abstract to be redone later."""
    print datetime.datetime.utcnow(),\
        '{} {}'.format(who, current_process().pid),\
        text


def _yield_blocks(container, block_size, iterations, offset=0):
    for block_no in range(iterations):
        block_start = offset + block_no * block_size
        yield container[block_start: block_start + block_size]


def _split_by_size(container, size, min_size_ratio=0.1):
    """Split jobs into the batches using preferable batch size."""
    blocks = len(container) // size

    for b in _yield_blocks(container, size, iterations=blocks - 1):
        yield b

    last_start = (blocks - 1) * size
    tail_block = container[blocks * size:]

    if len(tail_block) / size < min_size_ratio:
        yield container[last_start:]
    else:
        yield container[last_start: last_start + size]
        if tail_block:
            yield tail_block


def _split_by_workers(container, workers):
    """Split jobs into the batches using preferable numbner of workers."""
    standard_size, leftovers = divmod(len(container), workers)
    extended_size = standard_size + 1

    first = _yield_blocks(
        container, extended_size,
        iterations=leftovers
    )

    last = _yield_blocks(
        container, standard_size,
        iterations=workers - leftovers,
        offset=leftovers * extended_size
    )

    return chain(first, last)


def _indexed_batches(tasks, block_limit, workers):
    for batch in _split_by_size(tasks, block_limit):
        yield _split_by_workers(batch, workers)


def empty_shuffle(iterable):
    """No shuffle helper."""
    return iterable


def _dispatch_worker(tasks_conn, results_conn,
                     user_func, shuffle_func,
                     dispatcher_worker_num):
    # TODO: calculate maxsize more precisely
    queue = JoinableQueue(
        maxsize=10 * dispatcher_worker_num
    )

    pool = StandardPool(
        dispatcher_worker_num,
        initializer=_init_worker, initargs=(queue,)
    )

    collector_in, collector_out = Pipe(duplex=False)
    collector = OrderMerger(queue=queue,
                            out_conn=results_conn, in_conn=collector_in)
    collector.start()

    queue.close()

    if shuffle_func is None:
        shuffle_func = empty_shuffle

    try:
        while True:
            batch_tasks = marshal.loads(tasks_conn.recv_bytes())
            if batch_tasks is None:
                break

            batch_tasks = list(
                shuffle_func(enumerate(batch_tasks))
            )

            full_params = (
                (user_func, it)
                for it in batch_tasks
            )

            collector_out.send(len(batch_tasks))

            [
                pool.apply_async(_worker, (p,))
                for p in full_params
            ]

            del batch_tasks
    except Exception as ex:
        log_info('Dispatcher', 'finishing with ex: {}'.format(ex))
        collector.terminate()
        raise
    else:
        collector_out.send(0)
    finally:
        pool.close()
        pool.join()

        collector.join()


class BaseMerger(Process):
    """Base class to merge results form dispatcher' workers."""

    def __init__(self, queue, out_conn, in_conn, **kwargs):
        """Passive constructor."""
        super(BaseMerger, self).__init__(target=self._target, **kwargs)

        self.queue = queue
        self.out_conn = out_conn
        self.in_conn = in_conn

        self.results = None

    def pre_batch(self, batch_size):
        """Callback to prepare structures for a new batch of results."""
        self.results = []

    def merge_results(self, results):
        """Actual merge function."""
        self.results += results

    def _wait_for_batch_size(self):
        return self.in_conn.recv()

    def _target(self):
        try:
            stats = collections.Counter()

            batch_size = self._wait_for_batch_size()
            self.pre_batch(batch_size)
            task_count_down = batch_size

            while batch_size:
                message = self.queue.get()
                self.queue.task_done()

                message = marshal.loads(message)

                is_error, data = message
                if is_error:
                    raise Exception(data)

                results, _stats = data
                stats.update(_stats)

                self.merge_results(results)

                task_count_down -= 1
                if task_count_down <= 0:
                    self.out_conn.send(stats)
                    self.out_conn.send_bytes(marshal.dumps(self.results, 2))

                    batch_size = self._wait_for_batch_size()
                    self.pre_batch(batch_size)
                    task_count_down = batch_size
        except Exception as ex:
            print ex
            raise
        finally:
            self.queue.close()
            self.queue.join()


class OrderMerger(BaseMerger):
    """Class to merge results according to their task index."""

    def pre_batch(self, batch_size):
        """Create batch size of None' array."""
        self.results = [None] * batch_size

    def merge_results(self, results):
        """Place results according to their task index to the array."""
        for index, results_per_index in results:
            try:
                self.results[index].extend(results_per_index)
            except AttributeError:
                self.results[index] = results_per_index
            except IndexError:
                log_info('Merger', '{} {}'.format(
                    index, len(self.results)
                ))
                raise


class Multiprocessor(Process):
    """Manager process for task batching and dispatcher/mergers communication."""

    def __init__(self, dispatchers, workers_per_dispatcher,
                 main_connection_in, main_connection_out,
                 batch_size, **kwargs):
        """Prepare params and start a new process."""
        super(Multiprocessor, self).__init__(target=self._target, **kwargs)

        self.dispatchers_num = dispatchers
        self.workers_per_dispatcher = workers_per_dispatcher
        self.main_conn_in = main_connection_in
        self.batch_size = batch_size
        self.main_conn_out = main_connection_out

        self.start()

    def _init_dispatchers(self, user_func, shuffle_func):
        self.dispatchers = []
        self.dispatchers_conns = []
        self.results_conns = []
        for i in range(self.dispatchers_num):
            results_conn = Pipe(duplex=False)
            dispatcher_conn = Pipe(duplex=False)
            dispatcher = Process(
                target=_wrapped_dispatch_worker,
                args=(
                    dispatcher_conn[0], results_conn[1],
                    user_func, shuffle_func, self.workers_per_dispatcher
                )
            )
            dispatcher.start()

            self.dispatchers.append(dispatcher)
            self.dispatchers_conns.append(dispatcher_conn[1])
            self.results_conns.append(results_conn[0])

    def _target(self):
        stats = collections.Counter()

        user_func = self.main_conn_in.recv()
        shuffle_func = self.main_conn_in.recv()
        tasks = self.main_conn_in.recv()

        self._init_dispatchers(user_func, shuffle_func)

        # parts = self.batch_size // self.dispatchers_num
        parts = self.dispatchers_num
        if self.batch_size < parts * self.workers_per_dispatcher:
            partitions = _indexed_batches(tasks, self.batch_size, self.dispatchers_num)
        else:
            partitions = _indexed_batches(tasks, self.batch_size, parts)

        dispatcher_tasks = (
            tasks
            for b_collection in partitions
            for tasks in b_collection
        )

        dispatch_conns = cycle(enumerate(self.dispatchers_conns))
        result_conns = cycle(enumerate(self.results_conns))

        def dispatch(ind, conn, sub_tasks):
            conn.send_bytes(marshal.dumps(sub_tasks, 2))

        def recieve(ind, conn):
            stats.update(conn.recv())
            self.main_conn_out.send_bytes(conn.recv_bytes())

        try:
            for i in range(self.dispatchers_num):
                d_ind, d_conn = next(dispatch_conns)
                sub_tasks = next(dispatcher_tasks)
                dispatch(d_ind, d_conn, sub_tasks)

            for sub_tasks in dispatcher_tasks:
                recieve(*next(result_conns))

                d_ind, d_conn = next(dispatch_conns)
                dispatch(d_ind, d_conn, sub_tasks)

            for i in range(self.dispatchers_num):
                recieve(*next(result_conns))

            for dispatcher_conn in self.dispatchers_conns:
                dispatcher_conn.send_bytes(marshal.dumps(None, 2))

            self.main_conn_out.send_bytes(marshal.dumps(None, 2))
            self.main_conn_out.close()
        except:
            for dispatcher in self.dispatchers:
                dispatcher.terminate()
            raise
        finally:
            for dispatcher in self.dispatchers:
                dispatcher.join()

            self.dispatchers = None


class Pool(object):
    """
    Main thread multiprocessing Pool-like class.

    For now pool is available only for one-time map.
    """

    def __init__(self, dispatchers_num,
                 dispatcher_worker_num=None, batch_size=10 ** 6):
        """Start all necessary processes."""
        self.dispatchers_num = dispatchers_num
        if dispatcher_worker_num is None:
            dispatcher_worker_num = cpu_count() // dispatchers_num
        self.dispatcher_worker_num = dispatcher_worker_num
        read_conn, write_to_multiproc = Pipe(duplex=False)
        read_results, write_to_me = Pipe(duplex=False)
        self.multiproc_conn = write_to_multiproc
        self.results_conn = read_results
        self.multiprocessor = Multiprocessor(
            self.dispatchers_num, self.dispatcher_worker_num,
            read_conn, write_to_me, batch_size
        )

    def map(self, proc_func, tasks, shuffle_func=None):
        """Map function."""
        self.multiproc_conn.send(proc_func)
        self.multiproc_conn.send(shuffle_func)
        self.multiproc_conn.send(tasks)
        del tasks

        while True:
            batch_results = self.results_conn.recv_bytes()

            batch_results = marshal.loads(batch_results)
            if batch_results is None:
                # got signal to stop
                break

            for results_per_task in batch_results:
                yield results_per_task

        self.close()

    def close(self):
        """Close all connections and thus start demolishing the system."""
        self.results_conn.close()
        self.multiproc_conn.close()


def _init_worker(_queue):
    global queue
    queue = _queue


def _worker(item):
    global queue
    proc_func, params = item
    try:
        results = proc_func(params)
        queue.put(
            marshal.dumps(
                (False, results),
                2
            )
        )
    except:
        logger = log_to_stderr()
        logger.exception('Worker: exception')

        queue.put(
            marshal.dumps(
                (True, traceback.format_exc()),
                2
            )
        )


def _wrapped_dispatch_worker(*args):
    try:
        return _dispatch_worker(*args)
    except:
        print 'ERROR'
        traceback.print_exc()
