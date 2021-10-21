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

__all__ = ['getdminfo', 'getdesc', 'Table']


class VariableShapeArrayList(list):
    pass


def _peek(f, length):
    # Internal function used for debugging - show the next N bytes (and the
    # previous 8)
    pos = f.tell()
    f.seek(pos - 8)
    print(repr(f.read(8)), '  ', repr(f.read(length)))
    f.seek(pos)


def _peek_no_prefix(f, length):
    # Internal function used for debugging - show the next N bytes
    pos = f.tell()
    print(repr(f.read(length)))
    f.seek(pos)


class BaseCasaObject:
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
        if nbytes == 0:
            return
        bytes = f.read(nbytes - 4)
        b = EndianAwareFileHandle(BytesIO(bytes), f.endian, f.original_filename)
        if self:
            result = func(self, b, *args)
        else:
            result = func(b, *args)
        end = f.tell()
        # if end - start != nbytes:
        #     raise IOError('Function {0} read {1} bytes instead of {2}'
        #                   .format(func, end - start, nbytes))
        return result
    return wrapper



def check_type_and_version(f, name, versions):

    # HACK: sometimes the endian flag is not set correctly on f, and we need
    # to figure out why. In the mean time, we can tell the actual endianness
    # from the next byte, because we expect the next four bytes to be the length
    # of the name string, and this won't be ridiculously long.

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
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

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
    'string': ('String', read_string, '<U16'),
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
TO_DTYPE['string'] = '<U16'
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
        array = np.array([read_string(f, length_modifier=length_modifier) for i in range(nelem)])
        if nelem > 0:
            if max([len(s) for s in array]) < 16:  # HACK: only needed for getdesc comparisons
                array = array.astype('<U16')
    elif value_type == 'bool':
        array = np.unpackbits(np.frombuffer(f.read(int(np.ceil(nelem / 8)) * 8), dtype='uint8'), bitorder='little').astype(bool)[:nelem]
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


class Record(BaseCasaObject):

    @classmethod
    @with_nbytes_prefix
    def read(cls, f):
        self = cls()
        check_type_and_version(f, 'Record', 1)
        self.desc = RecordDesc.read(f)
        read_int32(f)  # Not sure what the following value is


class RecordDesc(BaseCasaObject):

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



class TableRecord(BaseCasaObject):

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


class Table(BaseCasaObject):

    @classmethod
    def read(cls, filename, endian='<'):
        """
        Read a CASA table - note that this is currently **experimental** and
        should not be used for production code.
        """

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

                    if isinstance(dm, (TiledCellStMan, TiledShapeStMan, StManAipsIO)):
                        endian = '>'

                    f = EndianAwareFileHandle(f_orig, endian, filename)

                    magic = f.read(4)
                    if magic != b'\xbe\xbe\xbe\xbe':
                        raise ValueError('Incorrect magic code: {0}'.format(magic))

                    dm.read_header(f)

        table._filename = filename

        return table

    def as_astropy_tables(self):

        # We now loop over columns and read the relevant data from each bucket.

        # TODO: refactor this to make it so that columns are read on request
        # instead of all in one go. The best way to do this might be to make a
        # table where each column is a dask array.

        coldesc = self.desc.column_description

        table_columns = OrderedDict()

        for colindex in range(len(coldesc)):

            # Find the data manager to use for the column as well as the
            # 'sequence number' - this is the value in e.g. table.f<seqnr>
            seqnr = self.column_set.columns[colindex].data.seqnr
            dm = self.column_set.data_managers[seqnr]

            # Each data manager might only handle one or a few of the columns.
            # It may internally have a list of the columns it deals with, so
            # we need to figure out what the column index is in that specific
            # data manager
            colindex_in_dm = 0
            for column in self.column_set.columns[:colindex]:
                if column.data.seqnr == seqnr:
                    colindex_in_dm += 1

            colname = coldesc[colindex].name

            if hasattr(dm, 'read_column'):
                coldata = dm.read_column(self._filename, seqnr, self.column_set.columns[colindex], coldesc[colindex], colindex_in_dm)
                if coldata is not None:
                    table_columns[colname] = coldata
            else:
                warnings.warn(f'Skipping column {colname} with data manager {dm.__class__.__name__}')

        # Some columns have variable shape - in this case we split the output
        # into several tables. We keep track of all the locations where the
        # shape changes and later combine that into a single list.

        split = None
        last_rows = {}

        for colname, data in table_columns.items():
            if isinstance(data, VariableShapeArrayList):
                if split is None:
                    split = set()
                split |= set([array.shape[0] for array in data])
                last_rows[colname] = np.cumsum([array.shape[0] for array in data])

        if split is None:
            return [AstropyTable(data=table_columns)]
        else:
            # Convert to a sorted list
            split = sorted(split)

            tables = []

            start = 0
            for end in split:

                columns_sub = OrderedDict()
                for colname, data in table_columns.items():
                    if isinstance(data, VariableShapeArrayList):
                        if len(data) == 0:
                            continue
                        index = np.searchsorted(last_rows[colname], start, side='right')
                        if index == 0:
                            offset = 0
                        else:
                            offset = last_rows[colname][index - 1]
                        columns_sub[colname] = data[index][start - offset:end - offset]
                    else:
                        columns_sub[colname] = data[start:end]

                tables.append(AstropyTable(data=columns_sub))

                start = end

            # For MS files, each table will likely have a unique DATA_DESC_ID so
            # for that special case we could have users select the required table
            # with this, but that could happen at a higher level.

            return tables

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


class TableDesc(BaseCasaObject):

    @classmethod
    @with_nbytes_prefix
    def read(cls, f, nrow):

        self = cls()

        check_type_and_version(f, 'TableDesc', 2)

        unknown1 = read_string(f)  # noqa
        unknown2 = read_string(f)  # noqa
        unknown3 = read_string(f)  # noqa

        self.keywords = TableRecord.read(f)
        self.private_keywords = TableRecord.read(f)

        self.ncol = read_int32(f)

        self.column_description = []

        for icol in range(self.ncol):
            self.column_description.append(ColumnDesc.read(f))

        return self


class StandardStMan(BaseCasaObject):

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

        version = check_type_and_version(f, 'StandardStMan', (2, 3))

        if version >= 3:
            self.big_endian = f.read(1) == b'\x01'  # noqa
        else:
            self.big_endian = True

        self.bucket_size = read_int32(f)
        self.number_of_buckets = read_int32(f)
        self.persistent_cache = read_int32(f)
        self.number_of_free_buckets = read_int32(f)
        self.first_free_bucket = read_int32(f)
        self.number_of_bucket_for_index = read_int32(f)
        self.first_index_bucket_number = read_int32(f)
        self.idx_bucket_offset = read_int32(f)
        self.last_string_bucket = read_int32(f)
        self.index_length = read_int32(f)
        self.number_indices = read_int32(f)

    def read_column(self, filename, seqnr, column, coldesc, colindex_in_dm):

        # Open the main file corresponding to the data manager
        fx_filename = os.path.join(filename, f'table.f{seqnr}')
        f = EndianAwareFileHandle(open(fx_filename, 'rb'), '>' if self.big_endian else '<', filename)

        # Open indirect array file if needed (sometimes arrays are stored
        # in these files).
        if os.path.exists(fx_filename + 'i'):
            fi = EndianAwareFileHandle(open(fx_filename + 'i', 'rb'), '>' if self.big_endian else '<', filename)
        else:
            fi = None

        # Variable length strings are stored in their own buckets which we cache
        # as needed.
        variable_string_buckets = {}
        next_variable_string_buckets = {}

        def _ensure_variable_string_bucket_loaded(f, vs_bucket_id):
            if vs_bucket_id in variable_string_buckets:
                return next_variable_string_buckets[vs_bucket_id]
            pos = f.tell()
            f.seek(512 + self.bucket_size * vs_bucket_id + 12)
            # For some reason, the next bucket index is stored in big endian
            next_vs_bucket_id = bytes_to_int32(f.read(4), '>')
            variable_string_buckets[vs_bucket_id] = f.read(self.bucket_size - 16)
            next_variable_string_buckets[vs_bucket_id] = next_vs_bucket_id
            f.seek(pos)
            return next_vs_bucket_id

        # To read in the SSMIndex we need to pre-load the index buckets. In
        # cases where the index is split over multiple buckets, the first four
        # bytes of the bucket indicate the index of the next index bucket to
        # read. This corresponds to the logic in SSMBase.readIndexBuckets:
        # https://github.com/casacore/casacore/blob/master/tables/DataMan/SSMBase.cc#L454
        # We pre-load the index buckets into a single BytesIO to make it look
        # contiguous.

        index_bytes = b''
        next_index_bucket = self.first_index_bucket_number
        remaining_index_length = self.index_length

        for bucket_id in range(self.number_of_bucket_for_index):

            bucket_start = 512 + next_index_bucket * self.bucket_size

            f.seek(bucket_start)

            # For some reason, the next bucket index is stored in big endian
            next_index_bucket = bytes_to_int32(f.read(4), '>')

            if self.idx_bucket_offset > 0:
                f.seek(bucket_start + self.idx_bucket_offset)
                index_bytes += f.read(remaining_index_length)
            elif remaining_index_length < self.bucket_size:
                f.seek(bucket_start + 8)
                index_bytes += f.read(remaining_index_length)
            else:
                f.seek(bucket_start + 8)
                index_bytes += f.read(self.bucket_size - 8)

        index_bytes = EndianAwareFileHandle(BytesIO(index_bytes[4:]), f.endian, f.original_filename)

        index = SSMIndex.read(index_bytes)

        if index.bucket_number.elements == []:  # empty table
            if coldesc.value_type in TO_DTYPE:
                return np.array([], dtype=TO_DTYPE[coldesc.value_type])
            else:
                return None

        shape = column.data.shape
        nelements = int(np.product(shape))

        data = []

        rows_in_bucket = np.diff(np.hstack([0, np.array(index.last_row.elements) + 1]))
        rows_in_bucket = {key: max(0, value) for (key, value) in zip(index.bucket_number.elements, rows_in_bucket)}

        for bucket_id in index.bucket_number.elements:

            # Find the starting position of the column in the bucket
            f.seek(512 + self.bucket_size * (bucket_id + self.column_index_map.elements[colindex_in_dm]) + self.column_offset.elements[colindex_in_dm])

            if coldesc.value_type == 'string':
                if coldesc.maxlen == 0:
                    subdata = []
                    for irow in range(rows_in_bucket[bucket_id]):
                        bytes = f.read(8)
                        length = read_int32(f)
                        if length <= 8:
                            subdata.append(bytes[:length])
                        else:
                            vs_bucket_id = bytes_to_int32(bytes[:4], f.endian)
                            offset = bytes_to_int32(bytes[4:], f.endian)
                            next_vs_bucket_id =_ensure_variable_string_bucket_loaded(f, vs_bucket_id)
                            bytes = variable_string_buckets[vs_bucket_id][offset:offset + length]
                            if len(bytes) < length:
                                _ensure_variable_string_bucket_loaded(f, next_vs_bucket_id)
                                bytes += variable_string_buckets[next_vs_bucket_id][:length - len(bytes)]
                            if coldesc.ndim != 0:
                                if coldesc.is_fixed_shape:
                                    n = np.product(coldesc.shape)
                                    pos = 0
                                else:
                                    n = bytes_to_int32(bytes[4:8], '>')
                                    pos = 12
                                strings = []
                                for i in range(n):
                                    l = bytes_to_int32(bytes[pos:pos + 4], '>')
                                    strings.append(bytes[pos + 4: pos + 4 + l])
                                    pos += 4 + l
                                if coldesc.is_fixed_shape:
                                    strings = np.reshape(strings, coldesc.shape)
                                bytes = strings
                            subdata.append(bytes)
                    data.append(np.array(subdata))
                else:
                    data.append(np.fromstring(f.read(coldesc.maxlen * rows_in_bucket[bucket_id]), dtype=f'S{coldesc.maxlen}'))
            elif coldesc.value_type == 'record':
                # TODO: determine how to handle this properly
                warnings.warn(f'Skipping column {coldesc.name} with type record')
                data = None
            else:
                if coldesc.is_direct or 'Scalar' in coldesc.stype:
                    data.append(read_as_numpy_array(f, coldesc.value_type, rows_in_bucket[bucket_id] * nelements, shape=(-1,) + shape[::-1]))
                else:
                    values = []
                    for irow in range(rows_in_bucket[bucket_id]):
                        offset = read_int64(f)
                        fi.seek(offset)
                        ndim = read_int32(fi)
                        subshape = []
                        for idim in range(ndim):
                            subshape.append(read_int32(fi))
                        size = int(np.product(subshape))
                        values.append(read_as_numpy_array(fi, coldesc.value_type, size, shape=subshape[::-1]))
                    data.append(np.array(values))
        if data:
            if data[0].ndim > 1:
                return np.vstack(data)
            else:
                return np.hstack(data)
        else:
            return None



class IncrementalStMan(BaseCasaObject):

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

        version = check_type_and_version(f, 'IncrementalStMan', (4, 5))

        if version >= 5:
            self.big_endian = f.read(1) == b'\x01'  # noqa
        else:
            self.big_endian = True

        self.bucket_size = read_int32(f)
        self.number_of_buckets = read_int32(f)
        self.persistent_cache = read_int32(f)

        # Unique nr for column in this storage manager
        self.unique_number_column = read_int32(f)

        if version > 1:
            self.number_of_free_buckets = read_int32(f)
            self.first_free_bucket = read_int32(f)

    def read_column(self, filename, seqnr, column, coldesc, colindex_in_dm):

        # Open the main file corresponding to the data manager
        fx_filename = os.path.join(filename, f'table.f{seqnr}')
        f = EndianAwareFileHandle(open(fx_filename, 'rb'), '>' if self.big_endian else '<', filename)

        # Open indirect array file if needed (sometimes arrays are stored
        # in these files).
        if os.path.exists(fx_filename + 'i'):
            fi = EndianAwareFileHandle(open(fx_filename + 'i', 'rb'), '>' if self.big_endian else '<', filename)
        else:
            fi = None

        # Start off by reading the bucket
        f.seek(512 + self.number_of_buckets * self.bucket_size + 4)
        index = ISMIndex.read(f)

        rows_in_bucket = np.diff(index.last_row.elements)
        rows_in_bucket = {key: value for (key, value) in zip(index.bucket_number.elements, rows_in_bucket)}

        n_rows = index.last_row.elements[-1]

        if n_rows > 0:

            data = []

            for bucket_id in index.bucket_number.elements:

                # Now move to the data location in the bucket
                f.seek(512 + bucket_id * self.bucket_size)

                # Read in length of data
                length = read_int32(f)

                # Read indices next to find out how many 'change' values there are
                f.seek(512 + bucket_id * self.bucket_size + length)

                for i in range(colindex_in_dm + 1):

                    n_changes = read_int32(f)

                    # Read in the indices
                    indices = np.frombuffer(f.read(n_changes * 4), dtype=f.endian + 'i4')

                    # Read in the offsets
                    offsets = np.frombuffer(f.read(n_changes * 4), dtype=f.endian + 'i4')

                # Now go back and read data
                f.seek(516 + bucket_id * self.bucket_size)

                values = []
                for off in offsets:
                    f.seek(516 + bucket_id * self.bucket_size + off)
                    if coldesc.is_direct or 'Scalar' in coldesc.stype:
                        values.append(read_as_numpy_array(f, coldesc.value_type, 1, length_modifier=-4))
                        subshape = []
                    else:
                        offset = read_int64(f)
                        fi.seek(offset)
                        ndim = read_int32(fi)
                        read_int32(fi)
                        subshape = []
                        for idim in range(ndim):
                            subshape.append(read_int32(fi))
                        size = int(np.product(subshape))
                        values.append(read_as_numpy_array(fi, coldesc.value_type, size, shape=subshape[::-1]))
                if subshape:
                    values = np.vstack(values)
                else:
                    values = np.hstack(values)

                # Now expand into full size array

                # https://github.com/dask/dask/issues/4389
                repeats = np.diff(np.hstack([indices, rows_in_bucket[bucket_id]]))
                data.append(np.repeat(values, repeats, axis=0))

            if data[0].ndim > 1:
                return np.vstack(data)
            else:
                return np.hstack(data)

        else:

            return np.array([], dtype=TO_DTYPE[coldesc.value_type])



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


class SSMIndex(BaseCasaObject):

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

        return self


class ISMIndex(BaseCasaObject):

    # https://github.com/casacore/casacore/blob/dbf28794ef446bbf4e6150653dbe404379a3c429/tables/DataMan/ISMIndex.cc#L51s

    @classmethod
    @with_nbytes_prefix
    def read(cls, f):
        self = cls()
        version = check_type_and_version(f, 'ISMIndex', 1)
        self.n_used = read_int32(f)
        if version > 1:
            self.last_row = Block.read(f, read_int64)
        else:
            self.last_row = Block.read(f, read_int32)
        self.bucket_number = Block.read(f, read_int32)
        return self


class TiledStMan(BaseCasaObject):


    @with_nbytes_prefix
    def read_header(self, f):

        version = check_type_and_version(f, 'TiledStMan', (1, 2))

        if version >= 2:
            self.big_endian = f.read(1) == b'\x01'  # noqa
        else:
            self.big_endian = True

        # TODO: Set endian flag on f here

        self.seqnr = read_int32(f)
        # if self.seqnr != 0:
        #     raise ValueError("Expected seqnr to be 0, got {0}".format(self.seqnr))

        self.nrows = read_int32(f)
        # if self.nrows != 1:
        #     raise ValueError("Expected nrows to be 1, got {0}".format(self.nrows))

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

        self.max_tsm_index = 0

        for tsm_index in range(self.nrfile):

            # The following flag seems to control whether or not the TSM file is
            # opened by CASA, and is probably safe to ignore here.
            flag = bool(f.read(1) == b'\x01')

            if not flag:
                continue

            self.max_tsm_index = tsm_index

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

        Record.read(f)

        flag = f.read(1)  # noqa

        ndim2 = read_int32(f)  # noqa

        self.cube_shape = read_iposition(f)
        self.tile_shape = read_iposition(f)

        unknown = read_int32(f)  # noqa
        unknown = read_int32(f)  # noqa

    def read_column(self, filename, seqnr, column, coldesc, colindex_in_dm):

        # chunkshape defines how the chunks (array subsets) are written to disk
        chunkshape = tuple(self.default_tile_shape)

        # the total shape defines the final output array shape
        if len(self.cube_shape) > 0:
            totalshape = self.cube_shape
        else:
            # FIXME: below is not the right default!
            totalshape = np.array(chunkshape)

        return self._read_tsm_file(filename, seqnr, coldesc, totalshape, chunkshape)

    def _read_tsm_file(self, filename, seqnr, coldesc, totalshape, chunkshape, tsm_index=0):

        totalshape = np.asarray(totalshape)
        chunkshape = np.asarray(chunkshape)
        chunksize = np.product(chunkshape)

        # Need to expose the following somehow
        target_chunksize = None
        memmap = False

        # the ratio between these tells you how many chunks must be combined
        # to create a final stack along each dimension
        stacks = np.ceil(totalshape / chunkshape).astype(int)

        dtype = TO_DTYPE[coldesc.value_type]

        if coldesc.value_type != 'bool':
            if self.big_endian:
                dtype = '>' + dtype
            else:
                dtype = '<' + dtype

        itemsize = np.dtype(dtype).itemsize

        chunkshape = tuple(int(x) for x in chunkshape)
        totalshape = tuple(int(x) for x in totalshape)

        # CASA chunks are typically too small to be efficient, so we use a larger
        # chunk size for dask and then tell CASAArrayWrapper about both the native
        # and target chunk size.
        # chunkshape = determine_optimal_chunkshape(totalshape, chunkshape)

        if target_chunksize is None:
            target_chunksize = 10000000

        if chunksize < target_chunksize:

            # Find optimal chunk - since we want to be efficient we want the new
            # chunks to be contiguous on disk so we first try and increase the
            # chunk size in x, then y, etc.

            chunkoversample = previous_chunkoversample = [1 for i in range(len(chunkshape))]

            finished = False
            for dim in range(len(chunkshape)):
                factors = [f for f in range(stacks[dim] + 1) if stacks[dim] % f == 0]
                for factor in factors:
                    chunkoversample[dim] = factor
                    if np.product(chunkoversample) * chunksize > target_chunksize:
                        chunkoversample = previous_chunkoversample
                        finished = True
                        break
                    previous_chunkoversample = chunkoversample.copy()
                if finished:
                    break

        else:

            chunkoversample = (1,) * len(chunkshape)

        chunkshape = [c * o for (c, o) in zip(chunkshape, chunkoversample)]

        # Create a wrapper that takes slices and returns the appropriate CASA data
        from .casa_dask import CASAArrayWrapper

        img_fn = os.path.join(filename, f'table.f{seqnr}_TSM{tsm_index}')

        wrapper = CASAArrayWrapper(img_fn, totalshape, chunkshape,
                                   chunkoversample=chunkoversample, dtype=dtype,
                                   itemsize=itemsize, memmap=memmap)

        # Convert to a dask array
        import uuid
        from dask.array import from_array
        dask_array = from_array(wrapper, name='CASA Data ' + str(uuid.uuid4()),
                                chunks=chunkshape[::-1])

        # Since the chunks may not divide the array exactly, all the chunks put
        # together may be larger than the array, so we need to get rid of any
        # extraneous padding.
        final_slice = tuple([slice(dim) for dim in totalshape[::-1]])

        array = dask_array[final_slice]

        return array.compute()


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

        super(TiledCellStMan, self).read_header(f)

    def read_column(self, *args, **kwargs):
        array = super(TiledCellStMan, self).read_column(*args, **kwargs)
        array = array.reshape((1,) + array.shape)
        return array


class TiledShapeStMan(TiledStMan):

    @classmethod
    def read(cls, f):
        self = cls()
        self.name = 'TiledShapeStMan'
        return self

    @with_nbytes_prefix
    def read_header(self, f):
        check_type_and_version(f, 'TiledShapeStMan', 1)
        super(TiledShapeStMan, self).read_header(f)
        self.default_tile_shape = read_iposition(f)
        self.number_used_row_map = read_int32(f)

        # The data might be split into multiple cubes (TSM<index> files). The
        # following three values help us piece these together into the main
        # hypercube. Each of the lists has length n_cubes where n_cubes is the
        # number of individual cubes.

        # The index of the last row in the final hypercube in each section. For
        # instance, [9, 19] means that the ten first rows (0-9) are in the first
        # subcube and the second set of ten rows (10-19) are in the second cube.
        self.last_row_abs = Block.read(f, read_int32)

        # The index of the cube in which the rows are stored - this is the value
        # used as a suffix in the TSM filename, e.g. TSM2
        self.cube_index = Block.read(f, read_int32)

        # The index of the last row of the subcube.
        self.last_row_sub = Block.read(f, read_int32)

    def read_column(self, filename, seqnr, column, coldesc, colindex_in_dm):

        # chunkshape defines how the chunks (array subsets) are written to disk
        chunkshape = list(self.default_tile_shape)

        # TODO: for now we assume that the cubes are in the right order in
        # self.cube_index and that self.last_row_abs is monotically increasing.
        # This assumption might actually be ok but we should check.

        arrays = []
        for tsm_index, row_index in zip(self.cube_index.elements, self.last_row_sub.elements):
            subcubeshape = chunkshape[:-1] + [row_index + 1]
            subcubeshape[1] *= tsm_index
            subchunkshape = chunkshape.copy()
            subchunkshape[1] *= tsm_index
            if subchunkshape[-1] > subcubeshape[-1]:
                subchunkshape[-1] = subcubeshape[-1]
            arrays.append(self._read_tsm_file(filename, seqnr, coldesc, subcubeshape, subchunkshape, tsm_index=tsm_index))

        return VariableShapeArrayList(arrays)


class StManColumnAipsIO(BaseCasaObject):

    @classmethod
    def read(cls, f, value_type):
        self = cls()
        read_int32(f)
        version = check_type_and_version(f, 'StManColumnAipsIO', 2)
        self.nr = read_int32(f)
        irow = 0
        self.values = []
        while irow < self.nr:
            nr = read_int32(f)
            nr = read_int32(f)
            self.values.append(read_as_numpy_array(f, value_type, nr))
            irow += nr
        self.values = np.hstack(self.values)
        return self


class StManAipsIO(BaseCasaObject):

    @classmethod
    def read(cls, f):
        self = cls()
        self.name = 'StManAipsIO'
        return self

    @with_nbytes_prefix
    def read_header(self, f):
        version = check_type_and_version(f, 'StManAipsIO', 2)
        if version > 1:
            self.name = read_string(f)
        self.seqnr = read_int32(f)
        self.unique_number = read_int32(f)
        self.nrow = read_int32(f)
        self.ncol = read_int32(f)
        self.value_types = [TYPES[read_int32(f)] for icol in range(self.ncol)]
        self.columns = [StManColumnAipsIO.read(f, self.value_types[icol]) for icol in range(self.ncol)]

    def read_column(self, filename, seqnr, column, coldesc, colindex_in_dm):
        return self.columns[colindex_in_dm].values


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
        super(TiledColumnStMan, self).read_header(f)


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


class ColumnSet(BaseCasaObject):

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
            elif name == 'StManAipsIO':
                dm_cls = StManAipsIO
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


class PlainColumn(BaseCasaObject):

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


class ArrayColumnData(BaseCasaObject):

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


class ScalarColumnData(BaseCasaObject):

    @classmethod
    def read(cls, f):

        self = cls()

        self.version = read_int32(f)
        self.seqnr = read_int32(f)
        self.shape = ()

        return self


class ColumnDesc(BaseCasaObject):

    @classmethod
    def read(cls, f):

        self = cls()

        unknown = read_int32(f)  # noqa

        stype, sversion = read_type(f)

        if not stype.startswith(('ScalarColumnDesc', 'ScalarRecordColumnDesc', 'ArrayColumnDesc')) or sversion != 1:
            raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

        # https://github.com/casacore/casacore/blob/dbf28794ef446bbf4e6150653dbe404379a3c429/tables/Tables/BaseColDesc.cc#L285

        self.stype = stype
        self.name = read_string(f)
        self.comment = read_string(f)
        self.data_manager_type = read_string(f).replace('Shape', 'Cell')
        self.data_manager_group = read_string(f)
        self.value_type = TYPES[read_int32(f)]

        self.option = read_int32(f)
        self.is_direct = self.option & 1 == 1
        self.is_undefined = self.option & 2 == 2
        self.is_fixed_shape = self.option & 4 == 4

        self.ndim = read_int32(f)
        if self.ndim != 0:
            self.shape = read_iposition(f)  # noqa
        self.maxlen = read_int32(f)
        self.keywords = TableRecord.read(f)

        version = read_int32(f)
        if 'ArrayColumnDesc' in stype:
            sw = f.read(1)
        else:
            if self.value_type in ('ushort', 'short'):
                default = f.read(2)
            elif self.value_type in ('uint', 'int', 'float'):
                default = f.read(4)
            elif self.value_type in ('double', 'complex'):
                default = f.read(8)
            elif self.value_type in ('dcomplex'):
                default = f.read(16)
            elif self.value_type == 'bool':
                default = f.read(1)
            elif self.value_type == 'string':
                default = read_string(f)
            elif self.value_type == 'record':
                default = f.read(8)
            else:
                raise NotImplementedError(f"Can't read default value for {self.value_type}")

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


        bucket['CubeShape'] = bucket['CellShape'] = dm.cube_shapes[0]
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
