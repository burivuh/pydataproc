from __future__ import division

import datetime
import random

from itertools import izip
from multiprocessing import Pool as StandardPool
from intensive_pool import Pool


def worker_for_new_pool(item):
    index, data = item
    count = 0
    for i in range(data):
        count += i
    return [(index, [(count, data)])], {'records': 1}


def common_worker(item):
    data = item
    count = 0
    for i in range(data):
        count += i
    return (count, data), {'records': 1}


def unbalanced_tasks(task_num, max_tasks_length):
    return [
        random.randint(min_task_length, max_tasks_length)
        for i in range(task_num)
    ]


def const_no_time(tasks, max_tasks_length):
    return [1] * len(tasks)


def const_long_time(tasks, max_tasks_length):
    return [max_tasks_length // 10] * len(tasks)


def linear_small_time(tasks, max_tasks_length):
    return [
        (t // 10) or 1
        for t in tasks
    ]


def linear_long_time(tasks, max_tasks_length):
    return [
        t
        for t in tasks
    ]


def capped_unbalanced_time(tasks, max_tasks_length):
    return [
        random.randint(min_task_length, t)
        for t in tasks
    ]


def get_avg(xs):
    return sum(xs) / len(xs)


if __name__ == '__main__':
    tryouts_num = 5
    worker_num = 7
    dispatcher_num = 6
    max_tasks_length = 10 ** 4
    min_task_length = 10
    task_num = 100 * (max_tasks_length - min_task_length)
    batch_size = task_num // 4

    get_tasks = unbalanced_tasks
    tasks = get_tasks(task_num // 100, max_tasks_length)

    to_test = Pool(dispatcher_num, worker_num, batch_size=batch_size)
    res1 = []
    for calcd in to_test.map(worker_for_new_pool, tasks, shuffle_func=None):
        res1.extend(calcd)
    to_test.close()

    to_test = Pool(dispatcher_num, worker_num, batch_size=batch_size)
    res2 = []
    for calcd in to_test.map(worker_for_new_pool, tasks, shuffle_func=sorted):
        res2.extend(calcd)
    to_test.close()

    common = []
    for calcd, stats in map(common_worker, tasks):
        common.append(calcd)

    assert tuple(res1) == tuple(res2) == tuple(common)

    for get_timing in (const_no_time, const_long_time,
                       linear_small_time, linear_long_time,
                       capped_unbalanced_time):
        print '=' * 40
        print get_tasks.__name__, get_timing.__name__

        results = [0] * tryouts_num
        for tryout in range(tryouts_num):
            p1 = Pool(dispatcher_num, worker_num, batch_size=batch_size)
            p2 = StandardPool(dispatcher_num * worker_num)

            tasks = get_tasks(task_num, max_tasks_length)
            proc_time = get_timing(tasks, max_tasks_length)

            t0 = datetime.datetime.utcnow()

            for _, t in izip(p1.map(worker_for_new_pool, tasks), proc_time):
                common_worker(t)

            t1 = datetime.datetime.utcnow()

            for _, t in izip(p2.map(common_worker, tasks, chunksize=1), proc_time):
                common_worker(t)

            t2 = datetime.datetime.utcnow()

            for _, t in izip(p2.map(common_worker, tasks), proc_time):
                common_worker(t)

            t3 = datetime.datetime.utcnow()

            d1 = t1 - t0
            d2 = t2 - t1
            d3 = t3 - t2
            results[tryout] = (
                d1.total_seconds(), d2.total_seconds(), d3.total_seconds()
            )

        d1, d2, d3 = map(get_avg, zip(*results))

        print d1 / d3, d2 / d3

        p1.close()
        p2.close()
