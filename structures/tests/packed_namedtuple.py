"""
Tests for packed_namedtuple module units.

You can run tests like this:
PYTHONPATH=$PYTHONPATH:. python2.7 structures/tests/packed_namedtuple.py
"""

from __future__ import division

import cPickle as pickle
import collections
import marshal
from operator import itemgetter
import sys
import timeit
import unittest

from structures.packed_namedtuple import packedtuple


class BaseTest(unittest.TestCase):
    """Parent class for every test here. Defines data examples and helpers."""

    def __init__(self, *args, **kwargs):
        """Initialise sample data to test."""
        super(BaseTest, self).__init__(*args, **kwargs)

        self.test_objects = (
            {
                'format': ['B', 'I', 'f'],
                'fields': {
                    'field_a': 10,
                    'field_b': 102,
                    'field_c': 1222.15
                }
            },
            {
                'format': ['I', 'I', 'I', 'B', 'f'],
                'fields': {
                    'field_a': 10,
                    'field_b': 102,
                    'field_c': 1222,
                    'field_d': 1,
                    'field_e': 12.15
                }
            },
            {
                'format': ['14s', 'I'],
                'fields': {
                    'field_a': 'a' * 14,
                    'field_b': 64253
                }
            }
        )

        self.float_precision = 2

    def _create_tested_types(self):
        return tuple(
            packedtuple(
                'PackedType%d' % i, obj['format'], sorted(obj['fields'].keys())
            )
            for i, obj in enumerate(self.test_objects)
        )

    def _create_common_types(self):
        return tuple(
            collections.namedtuple(
                'NamedTuple%i' % i, sorted(obj['fields'].keys())
            )
            for i, obj in enumerate(self.test_objects)
        )

    def setUp(self):
        """
        Default setUp method.

        Creates namedtuples and similar packed namedtuples.
        """
        self.packed_types = self._create_tested_types()
        self.named_tuples = self._create_common_types()
        # register types for pickle to find
        for cls in self.named_tuples:
            globals()[cls.__name__] = cls

        for cls in self.packed_types:
            globals()[cls.__name__] = cls

    def info(self, message):
        """Default console log print."""
        print message

    def prepare_value(self, value, precision):
        """Fix floats for testing."""
        if isinstance(value, (str, bytes)):
            return value
        return round(value, precision)


class BasePostInstanceTest(BaseTest):
    """Base class for tests creating instances of namedtuples."""

    marshal_protocol = 2

    def create_instances(self, types):
        """Create instances using **kwargs (by name)."""
        return tuple(
            tpe(**obj['fields'])
            for tpe, obj in zip(types, self.test_objects)
        )

    def create_instances_by_index(self, types):
        """Create instances using *args (param order)."""
        def get_sorted_values(obj):
            return zip(*sorted(obj['fields'].items()))[1]

        return tuple(
            tpe(*get_sorted_values(obj))
            for tpe, obj in zip(types, self.test_objects)
        )

    def setUp(self):
        """Creating instances for checking."""
        super(BasePostInstanceTest, self).setUp()

        self.tested_instances = self.create_instances(self.packed_types)
        self.common_instances = self.create_instances(self.named_tuples)


class CreationTest(BasePostInstanceTest):
    """Tests for creation."""

    def test_name_creation(self):
        """Check packed tuple valid creation by name."""
        tested = self.create_instances(self.packed_types)
        common = self.create_instances(self.named_tuples)

        for t1, t2 in zip(tested, common):
            t1 = (self.prepare_value(e, self.float_precision) for e in t1)
            t2 = (self.prepare_value(e, self.float_precision) for e in t2)
            self.assertEqual(tuple(t1), tuple(t2))

    def test_equal_creations(self):
        """Check equality of named and indexed creation."""
        by_name = self.create_instances(self.packed_types)
        by_index = self.create_instances_by_index(self.packed_types)
        for t1, t2 in zip(by_name, by_index):
            self.assertEqual(t1, t2)


class SerializeablenessTest(BasePostInstanceTest):
    """Testing if pickle/marshal can use packed named tuples."""

    def test_pickleableness(self):
        """Pickle objects and check their states."""
        for tested in self.tested_instances:
            r = pickle.loads(pickle.dumps(tested))
            self.assertEqual(r, tested)

    def test_marshalableness(self):
        """Pickle objects and check their states."""
        for tested in self.tested_instances:
            packed_array = marshal.loads(
                marshal.dumps(tested, self.marshal_protocol)
            )
            r = tested._restore(packed_array)
            self.assertEqual(r, tested)


class AccessTest(BasePostInstanceTest):
    """Class for random access testing."""

    def test_indexed(self):
        """Test indexed access."""
        for test_obj, tested, common in zip(self.test_objects,
                                            self.tested_instances,
                                            self.common_instances):
            for i in range(len(test_obj['fields'])):
                tested_value = self.prepare_value(
                    tested[i], self.float_precision
                )
                common_value = common[i]
                self.assertEqual(
                    tested_value, common_value,
                    'index {}: {} != {}'.format(
                        i, tested_value, common_value
                    )
                )

    def test_named(self):
        """Test named access."""
        for test_obj, tested, common in zip(self.test_objects,
                                            self.tested_instances,
                                            self.common_instances):
            for field_name in test_obj['fields']:
                tested_value = self.prepare_value(
                    getattr(tested, field_name),
                    self.float_precision
                )
                common_value = getattr(common, field_name)
                self.assertEqual(
                    tested_value, common_value,
                    'field {}: {} != {}'.format(
                        field_name, tested_value, common_value
                    )
                )

    def test_iteration(self):
        """Test iteration over packed tuple."""
        for tested, common in zip(self.tested_instances,
                                  self.common_instances):
            for tested_value, common_value in zip(tested, common):
                tested_value = self.prepare_value(
                    tested_value, self.float_precision
                )
                self.assertEqual(
                    tested_value, common_value,
                    '{} != {}'.format(tested_value, common_value)
                )


class MemoryTest(BasePostInstanceTest):
    """Class for memory testing."""

    def test_size(self):
        """Compare namedtuple and packed tuple RAM impact."""
        for tested, common in zip(self.tested_instances,
                                  self.common_instances):
            tested_size = sys.getsizeof(tested)
            common_size = sys.getsizeof(common)
            common_size += sum(map(sys.getsizeof, common))

            self.info(
                'Size: tested {}, common {}'.format(
                    tested_size, common_size
                )
            )

            self.assertLess(tested_size / common_size, 0.6)
            self.assertMore(tested_size / common_size, 0.5)


def timer(func):
    """Helper for timing function."""
    return min(timeit.repeat(func, number=1000, repeat=1000))


class CreationSpeedTest(BasePostInstanceTest):
    """Class for creation speed testing."""

    def test_named_creation(self):
        """Compare namedtuple and packedtuple instance creation by keywords."""
        tested_time = timer(
            lambda: self.create_instances(self.packed_types)
        )
        common_time = timer(
            lambda: self.create_instances(self.named_tuples)
        )
        self.info(
            'Named creation: tested {}, common {}'.format(
                tested_time, common_time
            )
        )
        self.assertLess(tested_time / common_time, 1.95)

    def test_indexed_creation(self):
        """Compare namedtuple and packedtuple instance creation by args."""
        tested_time = timer(
            lambda: self.create_instances_by_index(self.packed_types)
        )
        common_time = timer(
            lambda: self.create_instances_by_index(self.named_tuples)
        )
        self.info(
            'Indexed creation: tested {}, common {}'.format(
                tested_time, common_time
            )
        )
        self.assertLess(tested_time / common_time, 1.15)


class SerializationSpeedTest(BasePostInstanceTest):
    """Class for serialization/deserialization speeding tests."""

    def _pickle(self, seq):
        return map(pickle.dumps, seq)

    def test_pickle_serialization(self):
        """Compare namedtuple and packedtuple serialization speed."""
        tested_time = timer(
            lambda: self._pickle(self.tested_instances)
        )

        common_time = timer(
            lambda: self._pickle(self.common_instances)
        )

        self.info(
            'Serialization: tested {}, common {}'.format(
                tested_time, common_time
            )
        )
        self.assertLess(tested_time / common_time, 0.65)
        self.assertMore(tested_time / common_time, 0.55)

    def test_deserialization(self):
        """Compare namedtuple and packedtuple deserialization speed."""
        def deserialize(serialized):
            return [
                pickle.loads(value)
                for value in serialized
            ]

        serialized = self._pickle(self.tested_instances)
        tested_time = timer(
            lambda: deserialize(serialized)
        )

        serialized = self._pickle(self.common_instances)
        common_time = timer(
            lambda: deserialize(serialized)
        )
        self.info(
            'Deserialization: tested {}, common {}'.format(
                tested_time, common_time
            )
        )
        self.assertLess(tested_time / common_time, 0.8)
        self.assertMore(tested_time / common_time, 0.7)


class AccessSpeedTest(BasePostInstanceTest):
    """Class for access speed testing."""

    def test_last_index(self):
        """Compare last element access by index for named and packed tuples."""
        tested_time = timer(
            lambda: map(itemgetter(-1), self.tested_instances)
        )
        common_time = timer(
            lambda: map(itemgetter(-1), self.common_instances)
        )
        self.info(
            'Last index access: tested {}, common {}'.format(
                tested_time, common_time
            )
        )
        self.assertLess(tested_time / common_time, 5)

    def test_last_name(self):
        """Compare last element access by name for named and packed tuples."""
        for tested, common, initial in zip(self.tested_instances,
                                           self.common_instances,
                                           self.test_objects):
            field = sorted(initial['fields'])[-1]
            tested_time = timer(
                lambda: getattr(tested, field)
            )
            common_time = timer(
                lambda: getattr(common, field)
            )
            self.info(
                'Last name access: tested {}, common {}'.format(
                    tested_time, common_time
                )
            )
            self.assertLess(tested_time / common_time, 5.8)

    def test_iteration(self):
        """Compare iterating for named and packed tuples."""
        for tested, common in zip(self.tested_instances,
                                  self.common_instances):
            tested_time = timer(
                lambda: [x for x in tested]
            )
            common_time = timer(
                lambda: [x for x in common]
            )
            self.info(
                'Iteration: tested {}, common {}'.format(
                    tested_time, common_time
                )
            )
            self.assertLess(tested_time / common_time, 2.6)


if __name__ == '__main__':
    unittest.main()
