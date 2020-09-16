"""Module defining packed version for a namedtuple-like class."""

from functools import partial
import struct
import sys


class PackedNamedTuple(str):
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

    __slots__ = tuple()

    _unsupported_format_chars = frozenset('xpP@=<>!')

    @classmethod
    def setup(cls, format, fields):
        """Prepare class for serialization."""
        if cls._unsupported_format_chars.intersection(format):
            raise ValueError('These format chars: {} are not supported'.format(
                tuple(cls._unsupported_format_chars)
            ))

        cls.size = struct.calcsize(format)
        cls.fields_in_order = fields

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

    def __new__(cls, *args, **kwargs):
        """Support namedtuple interface. Packs content."""
        if kwargs:
            args = [
                kwargs[field]
                for field in cls.fields_in_order
            ]

        try:
            value = cls.whole_packer(*args)
        except:
            raise Exception(
                'Bad values: {} for format of: {}'.format(
                    args, type(cls).__name__
                )
            )
        return super(PackedNamedTuple, cls).__new__(cls, value)

    def __getitem__(self, index):
        """Implement indexed access to the fields."""
        return self.unpackers[index](self)[0]

    def __getslice__(self, start, end):
        """Implement slice through whole unpacking."""
        return self.whole_unpacker(self)[start:end]

    def __iter__(self):
        """Implement iteration through whole unpacking."""
        return iter(self.whole_unpacker(self))

    def __getattr__(self, field):
        """Implement named access to the fields."""
        try:
            return self.unpackers_by_name[field](self)[0]
        except KeyError:
            raise AttributeError(field)
        except Exception:
            raise AttributeError(field)

    @classmethod
    def _restore(cls, value):
        return _pickle_restore(cls, value)

    def __reduce__(self):
        """Pickle serialization awareness."""
        return (
            _pickle_restore,
            (
                self.__class__, str(self)
            )
        )


def _pickle_restore(cls, value):
    return super(PackedNamedTuple, cls).__new__(cls, value)


def packedtuple(name, format, fields, register_type=None):
    """
    Namedtuple-like factory function for a class creation.

    Creates and registers type for the pickle protocol.
    """
    t = type(name, (PackedNamedTuple,), {'__slots__': tuple()})

    t.setup(format, fields)

    # Used from a namedtuple code
    # For pickling to work, the __module__ variable needs to be set to the
    # frame where the named tuple is created.  Bypass this step in enviroments
    # where sys._getframe is not defined (Jython for example) or sys._getframe
    # is not defined for arguments greater than 0 (IronPython).
    try:
        t.__module__ = sys._getframe(1).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        pass

    if register_type:
        def serialize(obj):
            return obj.packed_array

        def deserialize(binary_record):
            return t(binary_record)

        register_type(t, serialize, deserialize)

    return t
