import os
import warnings
from collections import OrderedDict

import numpy as np

from dask import array as da

from astropy.table import Table as AstropyTable

from .core import (check_type_and_version, BaseCasaObject, with_nbytes_prefix,
                   read_string, read_int32, read_bool, read_iposition,
                   EndianAwareFileHandle, read_type, TYPES, read_array,
                   read_float32, read_float64, read_complex64, read_complex128)

from .data_managers import (StandardStMan, IncrementalStMan, TiledCellStMan,
                            TiledShapeStMan, TiledColumnStMan, StManAipsIO,
                            VariableShapeArrayList)

from .record import RecordDesc

from .dask_mixin import dask_to_mixin


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

        read_int32(f)
        if 'ArrayColumnDesc' in stype:
            sw = f.read(1)  # noqa
        else:
            if self.value_type in ('ushort', 'short'):
                self._default = f.read(2)
            elif self.value_type in ('uint', 'int', 'float'):
                self._default = f.read(4)
            elif self.value_type in ('double', 'complex'):
                self._default = f.read(8)
            elif self.value_type in ('dcomplex'):
                self._default = f.read(16)
            elif self.value_type == 'bool':
                self._default = f.read(1)
            elif self.value_type == 'string':
                self._default = read_string(f)
            elif self.value_type == 'record':
                self._default = f.read(8)
            else:
                raise NotImplementedError(f"Can't read default value for {self.value_type}")

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


def ensure_mixin_columns(columns):
    new_columns = {}
    for colname, column in columns.items():
        if len(column) == 0:
            raise Exception(f"Column {colname} is empty")
        if isinstance(column, da.Array):
            new_columns[colname] = dask_to_mixin(column)
        else:
            new_columns[colname] = column
    return new_columns


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
                last_rows[colname] = np.cumsum([array.shape[0] for array in data])
                split |= set(np.cumsum([array.shape[0] for array in data]))


        if split is None:
            return [AstropyTable(data=ensure_mixin_columns(table_columns))]
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

                tables.append(AstropyTable(data=ensure_mixin_columns(columns_sub)))

                start = end

            # For MS files, each table will likely have a unique DATA_DESC_ID so
            # for that special case we could have users select the required table
            # with this, but that could happen at a higher level.

            return tables

    @classmethod
    @with_nbytes_prefix
    def read_fileobj(cls, f):

        self = cls()

        check_type_and_version(f, 'Table', 2)

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
