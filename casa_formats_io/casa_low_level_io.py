# Pure Python + Numpy implementation of CASA's getdminfo() and getdesc()
# functions for reading metadata about .image files.

import os
import struct
from io import BytesIO
from collections import OrderedDict
from textwrap import indent

import numpy as np

__all__ = ['getdminfo', 'getdesc']

TYPES = ['bool', 'char', 'uchar', 'short', 'ushort', 'int', 'uint', 'float',
         'double', 'complex', 'dcomplex', 'string', 'table', 'arraybool',
         'arraychar', 'arrayuchar', 'arrayshort', 'arrayushort', 'arrayint',
         'arrayuint', 'arrayfloat', 'arraydouble', 'arraycomplex',
         'arraydcomplex', 'arraystr', 'record', 'other']


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
        b = EndianAwareFileHandle(BytesIO(f.read(nbytes - 4)), f.endian, f.original_filename)
        if self:
            result = func(self, b, *args)
        else:
            result = func(b, *args)
        end = f.tell()
        # print('-> ended {0} at {1}'.format(func, end))
        if end - start != nbytes:
            raise IOError('Function {0} read {1} bytes instead of {2}'
                          .format(func, end - start, nbytes))
        return result
    return wrapper


def read_bool(f):
    return f.read(1) == b'\x01'


def read_int32(f):
    return np.int32(struct.unpack(f.endian + 'i', f.read(4))[0])


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
    'int': ('Int', read_int32, int)
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
                f.read(4)
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

        self.desc = RecordDesc.read(f)

        unknown = read_int32(f)  # noqa

        self.values = []

        for name, rectype in zip(self.desc.names, self.desc.types):
            if rectype == 'bool':
                self.values.append(read_bool(f))
            elif rectype == 'int':
                self.values.append(int(read_int32(f)))
            elif rectype == 'uint':
                self.values.append(int(read_int32(f)))
            elif rectype == 'float':
                self.values.append(float(read_float32(f)))
            elif rectype == 'double':
                self.values.append(float(read_float64(f)))
            elif rectype == 'complex':
                self.values.append(complex(read_complex64(f)))
            elif rectype == 'dcomplex':
                self.values.append(complex(read_complex128(f)))
            elif rectype == 'string':
                self.values.append(read_string(f))
            elif rectype == 'table':
                self.values.append('Table: ' + os.path.abspath(os.path.join(f.original_filename, read_string(f))))
            elif rectype == 'arrayint':
                self.values.append(read_array(f, 'int'))
            elif rectype == 'arrayfloat':
                self.values.append(read_array(f, 'float'))
            elif rectype == 'arraydouble':
                self.values.append(read_array(f, 'double'))
            elif rectype == 'arraycomplex':
                self.values.append(read_array(f, 'complex'))
            elif rectype == 'arraydcomplex':
                self.values.append(read_array(f, 'dcomplex'))
            elif rectype == 'arraystr':
                self.values.append(read_array(f, 'string'))
            elif rectype == 'record':
                self.values.append(TableRecord.read(f))
            else:
                raise NotImplementedError("Support for type {0} in TableRecord not implemented".format(rectype))

        return self

    def as_dict(self):
        getdesc_dict = {}
        for name, value in zip(self.desc.names, self.values):
            if isinstance(value, TableRecord):
                getdesc_dict[name] = value.as_dict()
            else:
                getdesc_dict[name] = value
        return getdesc_dict


def check_type_and_version(f, name, versions):
    if np.isscalar(versions):
        versions = [versions]
    stype, sversion = read_type(f)
    if stype != name or sversion not in versions:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))
    return sversion


class Table(AutoRepr):

    @classmethod
    def read(cls, filename, endian='>'):

        with open(os.path.join(filename, 'table.dat'), 'rb') as f_orig:

            f = EndianAwareFileHandle(f_orig, '>', filename)

            magic = f.read(4)
            if magic != b'\xbe\xbe\xbe\xbe':
                raise ValueError('Incorrect magic code: {0}'.format(magic))

            table = cls.read_fileobj(f)

        if len(table.column_set.data_managers) > 1:
            raise NotImplementedError("Can't yet deal with tables with more than one data manager")

        dm = table.column_set.data_managers[0]

        f0_filename = os.path.join(filename, 'table.f0')

        if os.path.exists(f0_filename):

            with open(f0_filename, 'rb') as f_orig:

                f = EndianAwareFileHandle(f_orig, endian, filename)

                magic = f.read(4)
                if magic != b'\xbe\xbe\xbe\xbe':
                    raise ValueError('Incorrect magic code: {0}'.format(magic))

                dm.read_header(f)

        return table

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

        self.column_set = ColumnSet.read(f, ncol=self.desc.ncol)

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
            if icol > 0:
                read_int32(f)
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
        self.last_string_bucket = read_int32(f)
        self.index_length = read_int32(f)
        self.number_indices = read_int32(f)


class TiledStMan(AutoRepr):


    @with_nbytes_prefix
    def read_header(self, f):

        check_type_and_version(f, 'TiledStMan', 2)

        self.big_endian = f.read(1) == b'\x01'  # noqa

        self.seqnr = read_int32(f)
        if self.seqnr != 0:
            raise ValueError("Expected seqnr to be 0, got {0}".format(self.seqnr))

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
        if self.nrfile != 1:
            raise ValueError("Expected nrfile to be 1, got {0}".format(self.nrfile))

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
    def read(cls, f, ncol):

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

        data_manager_cls = []

        for i in range(self.nr):

            name = read_string(f)
            seqnr = read_int32(f)

            if name == 'StandardStMan':
                dm_cls = StandardStMan
            elif name == 'TiledCellStMan':
                dm_cls = TiledCellStMan
            else:
                raise NotImplementedError('Data manager {0} not supported'.format(name))

            data_manager_cls.append(dm_cls)

        self.columns = [PlainColumn.read(f) for index in range(ncol)]

        # Prepare data managers

        f.read(8)  # includes a length in bytes and bebebebe, need to check how this behaves when multiple DMs are present

        self.data_managers = []

        for i in range(self.nr):

            self.data_managers.append(data_manager_cls[i].read(f))

        return self


class PlainColumn(AutoRepr):

    @classmethod
    def read(cls, f):

        self = cls()

        version = read_int32(f)

        if version < 2:
            raise NotImplementedError('Support for PlainColumn version {0} not implemented'.format(version))

        self.name = read_string(f)

        self.data = ScalarColumnData.read(f)

        return self


class ScalarColumnData(AutoRepr):

    @classmethod
    def read(cls, f):

        self = cls()

        version = read_int32(f)
        self.seqnr = read_int32(f)

        return self


class ColumnDesc(AutoRepr):

    @classmethod
    def read(cls, f):

        self = cls()

        unknown = read_int32(f)  # noqa

        stype, sversion = read_type(f)

        if not stype.startswith(('ScalarColumnDesc', 'ArrayColumnDesc')) or sversion != 1:
            raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

        self.name = read_string(f)
        self.comment = read_string(f)
        self.data_manager_type = read_string(f).replace('Shape', 'Cell')
        self.data_manager_group = read_string(f)
        self.value_type = TYPES[read_int32(f)]
        self.option = read_int32(f)
        self.ndim = read_int32(f)
        if self.ndim > 0:
            self.ipos = read_iposition(f)  # noqa
        self.maxlen = read_int32(f)
        self.keywords = TableRecord.read(f)
        if self.value_type in ('ushort', 'short'):
            f.read(2)
        if self.value_type in ('uint', 'int', 'float', 'string'):
            f.read(4)
        elif self.value_type in ('double', 'complex'):
            f.read(8)
        elif self.value_type in ('dcomplex'):
            f.read(16)

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
