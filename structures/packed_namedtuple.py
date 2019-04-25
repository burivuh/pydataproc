"""Module defining packed version for a namedtuple-like class."""

from functools import partial
import struct


class PackedNamedTuple(object):
    """
    More memory-efficient tuple with a collections.namedtuple-like interface.

    Best use case: you're processing lists of namedtuples through
    different stages. On some stages you use one or two fields of these
    namedtuples, on the others - none, just passes these tuple data for
    the next stage. So basically, you do a lot of serialization/deserialization
    and not very actively access fields. If you use packed class -
    you use less memory (more workers?) AND can gain some performance
    at the same time.

    Designed to replace collections.namedtuple in the code, so it has a
    similar creation protocol and implement indexed and named access to
    the fields.
    Uses a struct module to work. Basically a wrapper on a binary string.
    Currently supports numerics and fixed-length strings.

    Advantages:
    - at least 10% more efficient memory-wise and can be much more efficient
    on a decent number of fields
    - much faster serialization/deserialization (as it is always
    "serialized")

    Disadvantages:
    - up to 6x times slower at random access
    - up to 3x times slower at iterating
    - up to 2x times slower instance creation
    """

    __slots__ = ('packed_array',)

    _unsupported_format_chars = frozenset('xpP@=<>!')

    @classmethod
    def setup(cls, format, fields):
        """Prepare class for serialization."""
        cls.fields_in_order = fields

        if cls._unsupported_format_chars.intersection(format):
            raise ValueError('Unsupported chars in the format string')

        cls.unpackers = [
            partial(
                struct.Struct('=' + fmt).unpack_from,
                offset=struct.calcsize('=' + ''.join(format[:i]))
            )
            for i, fmt in enumerate(format)
        ]

        cls.unpackers_by_name = {
            f: u
            for f, u in zip(fields, cls.unpackers)
        }

        serializer = struct.Struct('={}'.format(''.join(format)))
        cls.whole_packer = serializer.pack
        cls.whole_unpacker = serializer.unpack

    def __init__(self, *args, **kwargs):
        """Support namedtuple interface. Packs content."""
        if kwargs:
            args = [
                kwargs[field]
                for field in self.fields_in_order
            ]

        try:
            self.packed_array = self.whole_packer(*args)
        except:
            raise Exception(
                'Bad values: {} for format of: {}'.format(
                    args, type(self).__name__
                )
            )

    def __getitem__(self, index):
        """Implement indexed access to the fields."""
        return self.unpackers[index](self.packed_array)[0]

    def __getslice__(self, start, end):
        """Implement slice through whole unpacking."""
        return self.whole_unpacker(self.packed_array)[start:end]

    def __iter__(self):
        """Implement iteration through whole unpacking."""
        return iter(self.whole_unpacker(self.packed_array))

    def __getattr__(self, field):
        """Implement named access to the fields."""
        try:
            return self.unpackers_by_name[field](self.packed_array)[0]
        except KeyError:
            raise AttributeError(field)
        except Exception:
            raise AttributeError(field)

    def __eq__(self, other):
        """Implement == by comparing binary strings."""
        return self.packed_array == other.packed_array

    def __reduce__(self):
        """Pickle serialization awareness."""
        return (
            _pickle_restore,
            (
                self.__class__.__name__, None, self.packed_array
            )
        )


def _pickle_restore(name, fields, value):
    cls = _TYPE_BY_NAME_REGISTRY[name]
    instance = cls.__new__(cls)
    instance.packed_array = value
    return instance


_TYPE_BY_NAME_REGISTRY = {}


def create_namedtuple(name, format, fields, register_type=None):
    """
    Namedtuple-like factory function for a class creation.

    Creates and registers type for the pickle protocol.
    """
    t = type(name, (PackedNamedTuple,), {'__slots__': []})

    t.setup(format, fields)

    def serialize(obj):
        return obj.packed_array

    def deserialize(binary_record):
        instance = t.__new__(t)
        instance.packed_array = binary_record
        return instance

    if register_type:
        register_type(t, serialize, deserialize)

    _TYPE_BY_NAME_REGISTRY[name] = t

    return t
