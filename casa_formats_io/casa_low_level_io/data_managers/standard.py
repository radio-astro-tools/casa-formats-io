import os
import warnings
from math import prod
from io import BytesIO

import numpy as np

from ..core import (check_type_and_version, BaseCasaObject, with_nbytes_prefix,
                    read_string, read_int32, read_int64, read_as_numpy_array,
                    Block, EndianAwareFileHandle, TO_DTYPE, read_mapping,
                    bytes_to_int32)

__all__ = ['StandardStMan']


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
        nelements = int(prod(shape))

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
                            next_vs_bucket_id = _ensure_variable_string_bucket_loaded(f, vs_bucket_id)
                            bytes = variable_string_buckets[vs_bucket_id][offset:offset + length]
                            if len(bytes) < length:
                                _ensure_variable_string_bucket_loaded(f, next_vs_bucket_id)
                                bytes += variable_string_buckets[next_vs_bucket_id][:length - len(bytes)]
                            if coldesc.ndim != 0:
                                if coldesc.is_fixed_shape:
                                    n = prod(coldesc.shape)
                                    pos = 0
                                else:
                                    n = bytes_to_int32(bytes[4:8], '>')
                                    pos = 12
                                strings = []
                                for i in range(n):
                                    length = bytes_to_int32(bytes[pos:pos + 4], '>')
                                    strings.append(bytes[pos + 4: pos + 4 + length])
                                    pos += 4 + length
                                if coldesc.is_fixed_shape:
                                    strings = np.reshape(strings, coldesc.shape)
                                bytes = strings
                            subdata.append(bytes)
                    try:
                        data.append(np.array(subdata))
                    except ValueError:
                        data.append(np.array(subdata, dtype=object))
                else:
                    data.append(np.frombuffer(f.read(coldesc.maxlen * rows_in_bucket[bucket_id]), dtype=f'S{coldesc.maxlen}'))
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
                        size = int(prod(subshape))
                        values.append(read_as_numpy_array(fi, coldesc.value_type, size, shape=subshape[::-1]))
                    try:
                        data.append(np.array(values))
                    except ValueError:
                        data.append(np.array(values, dtype=object))
        if data:
            if data[0].ndim > 1:
                return np.vstack(data)
            else:
                return np.hstack(data)
        else:
            return None
