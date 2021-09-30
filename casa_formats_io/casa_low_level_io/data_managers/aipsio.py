import numpy as np

from ..core import (check_type_and_version, BaseCasaObject, with_nbytes_prefix,
                    read_string, read_int32, TYPES, read_as_numpy_array)

__all__ = ['StManAipsIO']


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


class StManColumnAipsIO(BaseCasaObject):

    @classmethod
    def read(cls, f, value_type):
        self = cls()
        read_int32(f)
        check_type_and_version(f, 'StManColumnAipsIO', 2)
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
