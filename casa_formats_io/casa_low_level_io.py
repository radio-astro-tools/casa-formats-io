# Pure Python + Numpy implementation of CASA's getdminfo() and getdesc()
# functions for reading metadata about .image files.

import os
import struct
import warnings
from io import BytesIO
from collections import OrderedDict
from textwrap import indent

import numpy as np

from astropy.table import Table as AstropyTable

__all__ = ['getdminfo', 'getdesc']

TYPES = ['bool', 'char', 'uchar', 'short', 'ushort', 'int', 'uint', 'float',
         'double', 'complex', 'dcomplex', 'string', 'table', 'arraybool',
         'arraychar', 'arrayuchar', 'arrayshort', 'arrayushort', 'arrayint',
         'arrayuint', 'arrayfloat', 'arraydouble', 'arraycomplex',
         'arraydcomplex', 'arraystr', 'record', 'other']


def peek(f, length):
    pos = f.tell()
    f.seek(pos - 8)
    print(repr(f.read(8)), '  ', repr(f.read(length)))
    f.seek(pos)


def peek_no_prefix(f, length):
    pos = f.tell()
    print(repr(f.read(length)))
    f.seek(pos)


class AutoRepr:
    def __repr__(self):
        from pprint import pformat
        return f'{self.__class__.__name__}' + pformat(self.__dict__)


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
        start = f.tell()
        nbytes = int(read_int32(f))
        # print('-> calling {0} with {1} bytes starting at {2}'.format(func, nbytes, start))
        if nbytes == 0:
            return
        bytes = f.read(nbytes - 4)
        if len(bytes) < nbytes - 4:
            bytes += b'\x00' * (nbytes - 4 - len(bytes))
        b = EndianAwareFileHandle(BytesIO(bytes), f.endian, f.original_filename)
        if self:
            result = func(self, b, *args)
        else:
            result = func(b, *args)
        end = f.tell()
        # print('-> ended {0} at {1}'.format(func, end))
        # if end - start != nbytes:
        #     raise IOError('Function {0} read {1} bytes instead of {2}'
        #                   .format(func, end - start, nbytes))
        return result
    return wrapper


def read_bool(f):
    return f.read(1) == b'\x01'


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


def read_string(f):
    value = read_int32(f)
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
    'string': ('String', read_string, '<U16'),
    'int': ('Int', read_int32, int),
    'uint': ('uInt', read_int32, int)
}


@with_nbytes_prefix
def read_array(f, arraytype):

    typerepr, reader, dtype = ARRAY_ITEM_READERS[arraytype]

    stype, sversion = read_type(f)

    if stype != f'Array<{typerepr}>' or sversion != 3:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

    ndim = read_int32(f)
    shape = [read_int32(f) for i in range(ndim)]
    size = read_int32(f)

    values = [reader(f) for i in range(size)]

    return np.array(values, dtype=dtype).reshape(shape)


def read_type(f):
    tp = read_string(f)
    version = read_int32(f)
    return tp, version


@with_nbytes_prefix
def read_record(f):
    check_type_and_version(f, 'Record', 1)
    RecordDesc.read(f)
    read_int32(f)  # Not sure what the following value is


class RecordDesc(AutoRepr):

    @classmethod
    @with_nbytes_prefix
    def read(cls, f):

        self = cls()

        check_type_and_version(f, 'RecordDesc', 2)

        nrec = read_int32(f)

        records = OrderedDict()

        self.names = []
        self.types = []

        for i in range(nrec):
            self.names.append(read_string(f))
            self.types.append(TYPES[read_int32(f)])
            # Here we don't actually load in the data for may of the types - hence
            # why we don't do anything with the values we read in.
            if self.types[-1] in ('bool', 'int', 'uint', 'float', 'double',
                                  'complex', 'dcomplex', 'string'):
                comment = read_string(f)
            elif self.types[-1] == 'table':
                f.read(8)
            elif self.types[-1].startswith('array'):
                read_iposition(f)
                f.read(4)
            elif self.types[-1] == 'record':
                RecordDesc.read(f)
                read_int32(f)
            else:
                raise NotImplementedError("Support for type {0} in RecordDesc not implemented".format(rectype))

        return self



class TableRecord(AutoRepr):

    @classmethod
    @with_nbytes_prefix
    def read(cls, f):

        self = cls()

        check_type_and_version(f, 'TableRecord', 1)

        desc = RecordDesc.read(f)

        unknown = read_int32(f)  # noqa

        self.values = {}

        for name, rectype in zip(desc.names, desc.types):
            if rectype == 'bool':
                self.values[name] = read_bool(f)
            elif rectype == 'int':
                self.values[name] = int(read_int32(f))
            elif rectype == 'uint':
                self.values[name] = int(read_int32(f))
            elif rectype == 'float':
                self.values[name] = float(read_float32(f))
            elif rectype == 'double':
                self.values[name] = float(read_float64(f))
            elif rectype == 'complex':
                self.values[name] = complex(read_complex64(f))
            elif rectype == 'dcomplex':
                self.values[name] = complex(read_complex128(f))
            elif rectype == 'string':
                self.values[name] = read_string(f)
            elif rectype == 'table':
                self.values[name] = 'Table: ' + os.path.abspath(os.path.join(f.original_filename, read_string(f)))
            elif rectype == 'arrayint':
                self.values[name] = read_array(f, 'int')
            elif rectype == 'arrayuint':
                self.values[name] = read_array(f, 'uint')
            elif rectype == 'arrayfloat':
                self.values[name] = read_array(f, 'float')
            elif rectype == 'arraydouble':
                self.values[name] = read_array(f, 'double')
            elif rectype == 'arraycomplex':
                self.values[name] = read_array(f, 'complex')
            elif rectype == 'arraydcomplex':
                self.values[name] = read_array(f, 'dcomplex')
            elif rectype == 'arraystr':
                self.values[name] = read_array(f, 'string')
            elif rectype == 'record':
                self.values[name] = TableRecord.read(f).values
            else:
                raise NotImplementedError("Support for type {0} in TableRecord not implemented".format(rectype))

        return self

    def as_dict(self):
        return self.values


def check_type_and_version(f, name, versions):
    if np.isscalar(versions):
        versions = [versions]
    stype, sversion = read_type(f)
    if stype != name or sversion not in versions:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))
    return sversion


TO_DTYPE = {}
TO_DTYPE['dcomplex'] = np.dtype('<c16')
TO_DTYPE['complex'] = np.dtype('<c8')
TO_DTYPE['double'] = np.dtype('<f8')
TO_DTYPE['float'] = np.dtype('<f4')
TO_DTYPE['short'] = np.dtype('<i2')
TO_DTYPE['int'] = np.dtype('<i4')
TO_DTYPE['bool'] = np.dtype('bool')


class Table(AutoRepr):

    @classmethod
    def read(cls, filename, endian='>'):

        with open(os.path.join(filename, 'table.dat'), 'rb') as f_orig:

            f = EndianAwareFileHandle(f_orig, '>', filename)

            magic = f.read(4)
            if magic != b'\xbe\xbe\xbe\xbe':
                raise ValueError('Incorrect magic code: {0}'.format(magic))

            table = cls.read_fileobj(f)

        for dm_index, dm in table.column_set.data_managers.items():

            fx_filename = os.path.join(filename, f'table.f{dm_index}')

            if os.path.exists(fx_filename):

                with open(fx_filename, 'rb') as f_orig:

                    if isinstance(dm, TiledShapeStMan):
                        endian = '>'

                    f = EndianAwareFileHandle(f_orig, endian, filename)

                    magic = f.read(4)
                    if magic != b'\xbe\xbe\xbe\xbe':
                        raise ValueError('Incorrect magic code: {0}'.format(magic))

                    dm.read_header(f)

        table._filename = filename

        return table

    def read_data(self):

        # We now loop over columns and read the relevant data from each bucket.

        # We can now loop over the data buckets and set up the table
        coldesc = self.desc.column_description

        # Variable length strings are stored in their own buckets which we cache
        # as needed.
        variable_string_buckets = {}

        t = AstropyTable()

        for colindex in range(len(coldesc)):

            seqnr = self.column_set.columns[colindex].data.seqnr
            dm = self.column_set.data_managers[seqnr]

            colindex_in_dm = 0
            for column in self.column_set.columns[:colindex]:
                if column.data.seqnr == seqnr:
                    colindex_in_dm += 1

            fx_filename = os.path.join(self._filename, f'table.f{seqnr}')
            f = EndianAwareFileHandle(open(fx_filename, 'rb'), '<', self._filename)

            # Start of buckets
            # Start off by reading index bucket. For very large tables there may be
            #  more than one index bucket in which case the code here will need to
            # be generalised.
            # if dm.number_of_bucket_for_index > 1:
            #     raise NotImplementedError("Can't yet read in data with more than one index bucket")


            if isinstance(dm, StandardStMan):

                f.seek(512 + dm.first_index_bucket_number * dm.bucket_size + dm.idx_bucket_offset + 4)

                index = SSMIndex.read(f)

                shape = self.column_set.columns[colindex].data.shape
                nelements = int(np.product(shape))
                value_type = coldesc[colindex].value_type

                data = []

                for bucket_id in index.bucket_number.elements:

                    # Find the starting position of the column in the bucket
                    f.seek(512 + dm.bucket_size * (bucket_id + dm.column_index_map.elements[colindex_in_dm]) + dm.column_offset.elements[colindex_in_dm])

                    if value_type == 'string':
                        # TODO: support shape != None for string columns
                        maxlen = coldesc[colindex].maxlen
                        if maxlen == 0:
                            for irow in range(index.rows_per_bucket):
                                bytes = f.read(8)
                                length = read_int32(f)
                                if length <= 8:
                                    data.append(bytes[:length])
                                else:
                                    vs_bucket_id = bytes_to_int32(bytes[:4], '<')
                                    offset = bytes_to_int32(bytes[4:], '<')
                                    if vs_bucket_id not in variable_string_buckets:
                                        pos = f.tell()
                                        f.seek(512 + dm.bucket_size * vs_bucket_id + 16)
                                        variable_string_buckets[vs_bucket_id] = f.read(dm.bucket_size - 16)
                                        f.seek(pos)
                                    data.append(variable_string_buckets[vs_bucket_id][offset:offset + length])
                        else:
                            data.append(np.fromstring(f.read(maxlen * index.rows_per_bucket), dtype=f'S{maxlen}'))
                    elif value_type in TO_DTYPE:
                        dtype = TO_DTYPE[value_type]
                        data.append(np.frombuffer(f.read(dtype.itemsize * index.rows_per_bucket * nelements), dtype=dtype).reshape((-1,) + shape))
                    else:
                        raise NotImplementedError(f"value type {value_type} not supported yet")
                if data:
                    t[coldesc[colindex].name] = np.hstack(data)[:index.last_row.elements[-1] + 1]
                else:
                    t[coldesc[colindex].name] = []
            elif isinstance(dm, IncrementalStMan):
                f.seek(512 + dm.number_of_buckets * dm.bucket_size + 4)
                index = ISMIndex.read(f)
                # TODO: Implement data reading
                warnings.warn(f'Igoring column {coldesc[colindex].name} with data manager {dm.__class__.__name__}')
            else:
                # TODO: Implement data reading
                pass
        return t


    @classmethod
    @with_nbytes_prefix
    def read_fileobj(cls, f):

        self = cls()

        version = check_type_and_version(f, 'Table', 2)

        self.nrow = read_int32(f)
        self.fmt = read_int32(f)  # noqa
        self.name = read_string(f)  # noqa

        # big_endian = fmt == 0  # noqa

        self.desc = TableDesc.read(f, self.nrow)

        self.column_set = ColumnSet.read(f, desc=self.desc)

        return self


class TableDesc(AutoRepr):

    @classmethod
    @with_nbytes_prefix
    def read(cls, f, nrow):

        self = cls()

        check_type_and_version(f, 'TableDesc', 2)

        unknown1 = read_int32(f)  # noqa
        unknown2 = read_int32(f)  # noqa
        unknown3 = read_string(f)  # noqa

        self.keywords = TableRecord.read(f)
        self.private_keywords = TableRecord.read(f)

        self.ncol = read_int32(f)

        self.column_description = []

        for icol in range(self.ncol):
            self.column_description.append(ColumnDesc.read(f))

        return self


class StandardStMan(AutoRepr):

    @classmethod
    @with_nbytes_prefix
    def read(cls, f):
        self = cls()
        check_type_and_version(f, 'SSM', 2)
        self.name = read_string(f)
        self.column_offset = Block.read(f, read_int32)
        self.column_index_map = Block.read(f, read_int32)
        self._fileobj = f
        return self

    @with_nbytes_prefix
    def read_header(self, f):

        # SSMBase::readHeader()
        # https://github.com/casacore/casacore/blob/d6da19830fa470bdd8434fd855abe79037fda78c/tables/DataMan/SSMBase.cc#L415

        check_type_and_version(f, 'StandardStMan', 3)

        self.big_endian = f.read(1) == b'\x01'  # noqa

        self.bucket_size = read_int32(f)
        self.number_of_buckets = read_int32(f)
        self.persistent_cache = read_int32(f)
        self.number_of_free_buckets = read_int32(f)
        self.first_free_bucket = read_int32(f)
        self.number_of_bucket_for_index = read_int32(f)
        self.first_index_bucket_number = read_int32(f)
        self.idx_bucket_offset = read_int32(f)
        if self.idx_bucket_offset == 0:
            # TODO: need to determine when this might not be 8
            # https://github.com/casacore/casacore/blob/dbf28794ef446bbf4e6150653dbe404379a3c429/tables/DataMan/SSMBase.cc#L459
            self.idx_bucket_offset = 8
        self.last_string_bucket = read_int32(f)
        self.index_length = read_int32(f)
        self.number_indices = read_int32(f)



class IncrementalStMan(AutoRepr):

    @classmethod
    @with_nbytes_prefix
    def read(cls, f):
        self = cls()
        check_type_and_version(f, 'ISM', 3)
        self.name = read_string(f)
        return self

    @with_nbytes_prefix
    def read_header(self, f):

        # SSMBase::readHeader()
        # https://github.com/casacore/casacore/blob/d6da19830fa470bdd8434fd855abe79037fda78c/tables/DataMan/SSMBase.cc#L415

        version = check_type_and_version(f, 'IncrementalStMan', 5)

        if version >= 5:
            self.big_endian = f.read(1) == b'\x01'  # noqa

        self.bucket_size = read_int32(f)
        self.number_of_buckets = read_int32(f)
        self.persistent_cache = read_int32(f)

        # Unique nr for column in this storage manager
        self.unique_number_column = read_int32(f)

        if version > 1:
            self.number_of_free_buckets = read_int32(f)
            self.first_free_bucket = read_int32(f)


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


class SSMIndex(AutoRepr):

    @classmethod
    @with_nbytes_prefix
    def read(cls, f):
        self = cls()
        check_type_and_version(f, 'SSMIndex', 1)
        self.n_used = read_int32(f)
        self.rows_per_bucket = read_int32(f)
        self.n_columns = read_int32(f)
        self.free_space = read_mapping(f, read_int32, read_int32)

        self.last_row = Block.read(f, read_int32)
        self.bucket_number = Block.read(f, read_int32)

        if self.bucket_number.elements == []:
            self.bucket_number.elements = list(range(self.n_used))

        return self


class ISMIndex(AutoRepr):

    # https://github.com/casacore/casacore/blob/dbf28794ef446bbf4e6150653dbe404379a3c429/tables/DataMan/ISMIndex.cc#L51s

    @classmethod
    @with_nbytes_prefix
    def read(cls, f):
        self = cls()
        version = check_type_and_version(f, 'ISMIndex', 1)
        self.n_used = read_int32(f)
        if version > 1:
            self.rows = Block.read(f, read_int64)
        else:
            self.rows = Block.read(f, read_int32)
        self.bucket_number = Block.read(f, read_int32)
        return self


class TiledStMan(AutoRepr):


    @with_nbytes_prefix
    def read_header(self, f):

        check_type_and_version(f, 'TiledStMan', 2)

        self.big_endian = f.read(1) == b'\x01'  # noqa

        self.seqnr = read_int32(f)
        # if self.seqnr != 0:
        #     raise ValueError("Expected seqnr to be 0, got {0}".format(self.seqnr))

        self.nrows = read_int32(f)
        if self.nrows != 1:
            raise ValueError("Expected nrows to be 1, got {0}".format(self.nrows))

        self.ncols = read_int32(f)
        if self.ncols != 1:
            raise ValueError("Expected ncols to be 1, got {0}".format(self.ncols))

        self.dtype = read_int32(f)
        self.column_name = read_string(f)
        self.max_cache_size = read_int32(f)
        self.ndim = read_int32(f)

        self.nrfile = read_int32(f)  # 1
        # if self.nrfile != 1:
        #     raise ValueError("Expected nrfile to be 1, got {0}".format(self.nrfile))

        # The following flag seems to control whether or not the TSM file is
        # opened by CASA, and is probably safe to ignore here.
        flag = bool(f.read(1))

        # The following two values are unknown, but are likely relevant when there
        # are more that one field in the image.

        mode = read_int32(f)
        unknown = read_int32(f)  # 0

        if mode == 1:
            self.total_cube_size = read_int32(f)
        elif mode == 2:
            self.total_cube_size = read_int64(f)
        else:
            raise ValueError('Unexpected value {0} at position {1}'.format(mode, f.tell() - 8))

        unknown = read_int32(f)  # 1
        unknown = read_int32(f)  # 1

        read_record(f)

        flag = f.read(1)  # noqa

        ndim2 = read_int32(f)  # noqa

        self.cube_shape = read_iposition(f)
        self.tile_shape = read_iposition(f)

        unknown = read_int32(f)  # noqa
        unknown = read_int32(f)  # noqa


class TiledCellStMan(TiledStMan):

    @classmethod
    def read(cls, f):
        self = cls()
        self.name = 'TiledCellStMan'
        return self

    @with_nbytes_prefix
    def read_header(self, f):

        # The code in this function corresponds to TiledStMan::headerFileGet
        # https://github.com/casacore/casacore/blob/75b358be47039250e03e5042210cbc60beaaf6e4/tables/DataMan/TiledStMan.cc#L1086

        check_type_and_version(f, 'TiledCellStMan', 1)

        self.default_tile_shape = read_iposition(f)

        super().read_header(f)


class TiledShapeStMan(TiledStMan):

    @classmethod
    def read(cls, f):
        self = cls()
        self.name = 'TiledShapeStMan'
        return self

    @with_nbytes_prefix
    def read_header(self, f):
        check_type_and_version(f, 'TiledShapeStMan', 1)
        # super().read_header(f)


class TiledColumnStMan(TiledStMan):

    @classmethod
    def read(cls, f):
        self = cls()
        self.name = 'TiledColumnStMan'
        return self

    @with_nbytes_prefix
    def read_header(self, f):
        check_type_and_version(f, 'TiledColumnStMan', 1)
        self.default_tile_shape = read_iposition(f)
        super().read_header(f)


class Block(AutoRepr):

    @classmethod
    def read(cls, f, func):
        self = cls()
        self.nr = read_int32(f)
        self.name = read_string(f)
        self.version = read_int32(f)
        self.size = read_int32(f)
        self.elements = [func(f) for i in range(self.size)]
        return self


class ColumnSet(AutoRepr):

    @classmethod
    def read(cls, f, desc):

        self = cls()

        version = read_int32(f)  # can be negative
        # See full logic in ColumnSet.getFile
        version = -version

        if version != 2:
            raise NotImplementedError('Support for ColumnSet version {0} not implemented'.format(version))

        self.nrow = read_int32(f)
        self.nrman = read_int32(f)
        self.nr = read_int32(f)

        # Construct data managers

        data_manager_cls = OrderedDict()

        for i in range(self.nr):

            name = read_string(f)
            seqnr = read_int32(f)

            if name == 'StandardStMan':
                dm_cls = StandardStMan
            elif name == 'IncrementalStMan':
                dm_cls = IncrementalStMan
            elif name == 'TiledCellStMan':
                dm_cls = TiledCellStMan
            elif name == 'TiledShapeStMan':
                dm_cls = TiledShapeStMan
            elif name == 'TiledColumnStMan':
                dm_cls = TiledColumnStMan
            else:
                raise NotImplementedError('Data manager {0} not supported'.format(name))

            data_manager_cls[seqnr] = dm_cls

        self.columns = [PlainColumn.read(f, ndim=desc.column_description[index].ndim) for index in range(desc.ncol)]

        # Prepare data managers

        f.read(8)  # includes a length in bytes and bebebebe, need to check how this behaves when multiple DMs are present

        self.data_managers = OrderedDict()

        for seqnr in data_manager_cls:
            self.data_managers[seqnr] = data_manager_cls[seqnr].read(f)
            f.read(8)

        return self


class PlainColumn(AutoRepr):

    @classmethod
    def read(cls, f, ndim):

        self = cls()

        version = read_int32(f)

        if version < 2:
            raise NotImplementedError('Support for PlainColumn version {0} not implemented'.format(version))

        self.name = read_string(f)

        if ndim != 0:
            self.data = ArrayColumnData.read(f)
        else:
            self.data = ScalarColumnData.read(f)

        return self


class ArrayColumnData(AutoRepr):

    @classmethod
    def read(cls, f):

        self = cls()

        self.version = read_int32(f)
        self.seqnr = read_int32(f)

        has_shape = read_bool(f)

        if has_shape:
            self.shape = tuple(read_iposition(f).tolist())
        else:
            self.shape = ()

        return self


class ScalarColumnData(AutoRepr):

    @classmethod
    def read(cls, f):

        self = cls()

        self.version = read_int32(f)
        self.seqnr = read_int32(f)
        self.shape = ()

        return self


class ColumnDesc(AutoRepr):

    @classmethod
    def read(cls, f):

        self = cls()

        unknown = read_int32(f)  # noqa

        stype, sversion = read_type(f)

        if not stype.startswith(('ScalarColumnDesc', 'ArrayColumnDesc')) or sversion != 1:
            raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

        # https://github.com/casacore/casacore/blob/dbf28794ef446bbf4e6150653dbe404379a3c429/tables/Tables/BaseColDesc.cc#L285

        self.name = read_string(f)
        self.comment = read_string(f)
        self.data_manager_type = read_string(f).replace('Shape', 'Cell')
        self.data_manager_group = read_string(f)
        self.value_type = TYPES[read_int32(f)]
        self.option = read_int32(f)
        self.ndim = read_int32(f)
        if self.ndim != 0:
            self.ipos = read_iposition(f)  # noqa
        self.maxlen = read_int32(f)
        self.keywords = TableRecord.read(f)

        version = read_int32(f)
        if 'ArrayColumnDesc' in stype:
            sw = f.read(1)
        else:
            if self.value_type in ('ushort', 'short'):
                default = f.read(2)
            if self.value_type in ('uint', 'int', 'float'):
                default = f.read(4)
            elif self.value_type in ('double', 'complex'):
                default = f.read(8)
            elif self.value_type in ('dcomplex'):
                default = f.read(16)
            elif self.value_type == 'bool':
                default = f.read(1)
            elif self.value_type == 'string':
                default = read_string(f)
            else:
                raise NotImplementedError(f"Can't read default value for {self.value_type}")

        pos = f.tell()
        f.seek(pos)


        return self



def getdminfo(filename, endian='>'):
    """
    Return the same output as CASA's getdminfo() function, namely a dictionary
    with metadata about the .image file, parsed from the ``table.f0`` file.
    """

    table = Table.read(filename, endian=endian)

    colset = table.column_set
    dm = colset.data_managers[0]

    dminfo = {}

    if isinstance(dm, StandardStMan):

        dminfo['COLUMNS'] = np.array(sorted(col.name for col in colset.columns), '<U16')
        dminfo['NAME'] = dm.name
        dminfo['SEQNR'] = 0
        dminfo['TYPE'] = 'StandardStMan'

    dminfo['SPEC'] = {}

    if isinstance(dm, StandardStMan):

        dminfo['SPEC']['BUCKETSIZE'] = dm.bucket_size
        dminfo['SPEC']['IndexLength'] = dm.index_length
        dminfo['SPEC']['MaxCacheSize'] = dm.persistent_cache  # NOTE: not sure if correct
        dminfo['SPEC']['PERSCACHESIZE'] = dm.persistent_cache

    elif isinstance(dm, TiledCellStMan):

        dminfo['SPEC']['DEFAULTTILESHAPE'] = dm.default_tile_shape
        dminfo['SEQNR'] = dm.seqnr
        dminfo['SPEC']['SEQNR'] = dm.seqnr

        dminfo['COLUMNS'] = np.array([dm.column_name], dtype='<U16')
        dminfo['NAME'] = dm.column_name

        dminfo['SPEC']['MAXIMUMCACHESIZE'] = dm.max_cache_size
        dminfo['SPEC']['MaxCacheSize'] = dm.max_cache_size


        bucket = dminfo['SPEC']['HYPERCUBES'] = {}
        bucket = dminfo['SPEC']['HYPERCUBES']['*1'] = {}


        bucket['CubeShape'] = bucket['CellShape'] = dm.cube_shape
        bucket['TileShape'] = dm.tile_shape
        bucket['ID'] = {}
        bucket['BucketSize'] = int(dm.total_cube_size /
                                    np.product(np.ceil(bucket['CubeShape'] / bucket['TileShape'])))

        dminfo['TYPE'] = 'TiledCellStMan'

    return {'*1': dminfo}


def getdesc(filename, endian='>'):
    """
    Return the same output as CASA's getdesc() function, namely a dictionary
    with metadata about the .image file, parsed from the ``table.dat`` file.
    """

    table = Table.read(filename, endian=endian)

    coldesc = table.desc.column_description

    desc = {}
    for column in coldesc:
        desc[column.name] = {'comment': column.comment,
                            'dataManagerGroup': table.column_set.data_managers[0].name,
                            'dataManagerType': column.data_manager_type,
                            'keywords': column.keywords.as_dict(),
                            'maxlen': column.maxlen,
                            'option': column.option,
                            'valueType': column.value_type}
    desc['_keywords_'] = table.desc.keywords.as_dict()
    desc['_private_keywords_'] = table.desc.private_keywords.as_dict()
    desc['_define_hypercolumn_'] = {}

    return desc
