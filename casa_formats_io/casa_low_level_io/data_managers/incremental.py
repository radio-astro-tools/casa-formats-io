import os
from math import prod

import numpy as np

from ..core import (check_type_and_version, BaseCasaObject, with_nbytes_prefix,
                    read_string, read_int32, read_int64, read_as_numpy_array,
                    Block, EndianAwareFileHandle, TO_DTYPE)

__all__ = ['IncrementalStMan']


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
                        size = int(prod(subshape))
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
