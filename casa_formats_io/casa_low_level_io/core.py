import struct
import warnings
from io import BytesIO

import numpy as np

TYPES = ['bool', 'char', 'uchar', 'short', 'ushort', 'int', 'uint', 'float',
         'double', 'complex', 'dcomplex', 'string', 'table', 'arraybool',
         'arraychar', 'arrayuchar', 'arrayshort', 'arrayushort', 'arrayint',
         'arrayuint', 'arrayfloat', 'arraydouble', 'arraycomplex',
         'arraydcomplex', 'arraystr', 'record', 'other']


class BaseCasaObject:
    def __repr__(self):
        from pprint import pformat
        return f'{self.__class__.__name__}' + pformat(self.__dict__)


class Block(BaseCasaObject):

    @classmethod
    def read(cls, f, func):
        self = cls()
        self.nr = read_int32(f)
        self.name = read_string(f)
        self.version = read_int32(f)
        self.size = read_int32(f)
        self.elements = [func(f) for i in range(self.size)]
        return self


class EndianAwareFileHandle:

    def __init__(self, file_handle, endian, original_filename):
        self.file_handle = file_handle
        self.endian = endian
        self.original_filename = original_filename

    def read(self, n=None):
        return self.file_handle.read(n)

    def tell(self):
        return self.file_handle.tell()

    def seek(self, n):
        return self.file_handle.seek(n)


def with_nbytes_prefix(func):
    def wrapper(*args):
        if hasattr(args[0], 'tell'):
            self = None
            f = args[0]
            args = args[1:]
        else:
            self = args[0]
            f = args[1]
            args = args[2:]
        # start = f.tell()
        nbytes = int(read_int32(f))
        if nbytes == 0:
            return
        bytes = f.read(nbytes - 4)
        b = EndianAwareFileHandle(BytesIO(bytes), f.endian,
                                  f.original_filename)
        if self:
            result = func(self, b, *args)
        else:
            result = func(b, *args)
        # end = f.tell()
        # if end - start != nbytes:
        #     raise IOError('Function {0} read {1} bytes instead of {2}'
        #                   .format(func, end - start, nbytes))
        return result
    return wrapper


def check_type_and_version(f, name, versions):

    # HACK: sometimes the endian flag is not set correctly on f, and we need to
    # figure out why. In the mean time, we can tell the actual endianness from
    # the next byte, because we expect the next four bytes to be the length of
    # the name string, and this won't be ridiculously long.

    start = f.tell()
    next = f.read(1)
    if next == b'\x00':
        actual_endian = '>'
    else:
        actual_endian = '<'
    f.seek(start)

    if actual_endian != f.endian:
        warnings.warn(f'Endianness of {name} did not match endianness of file'
                      'handle, correcting')
        f.endian = actual_endian

    if np.isscalar(versions):
        versions = [versions]
    stype, sversion = read_type(f)
    if stype != name or sversion not in versions:
        raise NotImplementedError('Support for {0} version {1} not implemented'
                                  .format(stype, sversion))

    return sversion


def read_bool(f):
    return f.read(1) == b'\x01'


def read_int16(f):
    return np.int16(struct.unpack(f.endian + 'h', f.read(2))[0])


def read_int32(f):
    return np.int32(struct.unpack(f.endian + 'i', f.read(4))[0])


def bytes_to_int32(bytes, endian):
    return np.int32(struct.unpack(endian + 'i', bytes)[0])


def read_int64(f):
    return np.int64(struct.unpack(f.endian + 'q', f.read(8))[0])


def read_float32(f):
    return np.float32(struct.unpack(f.endian + 'f', f.read(4))[0])


def read_float64(f):
    return np.float64(struct.unpack(f.endian + 'd', f.read(8))[0])


def read_complex64(f):
    return np.complex64(read_float32(f) + 1j * read_float32(f))


def read_complex128(f):
    return np.complex128(read_float64(f) + 1j * read_float64(f))


def read_string(f, length_modifier=0):
    value = read_int32(f) + length_modifier
    return f.read(int(value)).replace(b'\x00', b'').decode('ascii')


@with_nbytes_prefix
def read_iposition(f):
    check_type_and_version(f, 'IPosition', 1)
    nelem = read_int32(f)
    return np.array([read_int32(f) for i in range(nelem)], dtype=int)


ARRAY_ITEM_READERS = {
    'float': ('float', read_float32, np.float32),
    'double': ('double', read_float64, np.float64),
    'dcomplex': ('void', read_complex128, np.complex128),
    'string': ('String', read_string, 'U'),
    'int': ('Int', read_int32, int),
    'uint': ('uInt', read_int32, int)
}


TO_DTYPE = {}
TO_DTYPE['dcomplex'] = 'c16'
TO_DTYPE['complex'] = 'c8'
TO_DTYPE['double'] = 'f8'
TO_DTYPE['float'] = 'f4'
TO_DTYPE['int'] = 'i4'
TO_DTYPE['uint'] = 'u4'
TO_DTYPE['short'] = 'i2'
TO_DTYPE['string'] = 'U'
TO_DTYPE['bool'] = 'bool'
TO_DTYPE['record'] = 'O'

TO_TYPEREPR = {}
TO_TYPEREPR['dcomplex'] = 'void'
TO_TYPEREPR['double'] = 'double'
TO_TYPEREPR['float'] = 'float'
TO_TYPEREPR['int'] = 'Int'
TO_TYPEREPR['uint'] = 'uInt'
TO_TYPEREPR['string'] = 'String'


def read_as_numpy_array(f, value_type, nelem, shape=None, length_modifier=0):
    """
    Read the next 'nelem' values as a Numpy array
    """
    if value_type == 'string':
        array = np.array([read_string(f, length_modifier=length_modifier)
                          for i in range(nelem)])
    elif value_type == 'bool':
        length = int(np.ceil(nelem / 8)) * 8
        array = np.unpackbits(np.frombuffer(f.read(length), dtype='uint8'),
                              bitorder='little').astype(bool)[:nelem]
    elif value_type in TO_DTYPE:
        dtype = np.dtype(f.endian + TO_DTYPE[value_type])
        array = np.frombuffer(f.read(int(nelem * dtype.itemsize)), dtype=dtype)
    else:
        raise NotImplementedError(f"Can't read in data of type {value_type}")
    if shape is not None:
        array = array.reshape(shape)
    return array


@with_nbytes_prefix
def read_array(f, arraytype):

    typerepr = TO_TYPEREPR[arraytype]

    check_type_and_version(f, f'Array<{typerepr}>', 3)

    ndim = read_int32(f)
    shape = [read_int32(f) for i in range(ndim)]
    size = read_int32(f)

    return read_as_numpy_array(f, arraytype, size, shape=shape)


def read_type(f):
    tp = read_string(f)
    version = read_int32(f)
    return tp, version


@with_nbytes_prefix
def read_mapping(f, key_reader, value_reader):
    check_type_and_version(f, 'SimpleOrderedMap', 1)
    pos = f.tell()
    f.seek(pos)
    read_int32(f)  # ignored
    nr = read_int32(f)
    read_int32(f)  # ignored
    m = {}
    for i in range(nr):
        key = key_reader(f)
        value = value_reader(f)
        m[key] = value
    return m
