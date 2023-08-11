import os
from math import prod
from collections import defaultdict

import numpy as np

import dask.array as da

from ..core import (check_type_and_version, BaseCasaObject, with_nbytes_prefix,
                    read_string, read_int32, read_int64, read_iposition,
                    TO_DTYPE, Block)

from ..record import Record

__all__ = ['VariableShapeArrayList', 'TiledStMan', 'TiledCellStMan',
           'TiledShapeStMan', 'TiledColumnStMan']


class VariableShapeArrayList(list):
    pass


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

        for itsm in range(self.nrfile):

            # The following flag seems to control whether or not the TSM file is
            # opened by CASA, and is probably safe to ignore here.
            flag = bool(f.read(1) == b'\x01')

            if not flag:
                continue

            self.max_tsm_index = itsm

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

        self.cube_shapes = []
        self.tile_shapes = []

        for itsm in range(self.nrfile):

            unknown = read_int32(f)  # 1

            Record.read(f)

            flag = f.read(1)  # noqa

            ndim2 = read_int32(f)  # noqa

            self.cube_shapes.append(read_iposition(f))
            self.tile_shapes.append(read_iposition(f))

            unknown = read_int32(f)  # noqa
            unknown = read_int32(f)  # noqa

    def read_column(self, filename, seqnr, column, coldesc, colindex_in_dm):

        # chunkshape defines how the chunks (array subsets) are written to disk
        chunkshape = tuple(self.tile_shapes[0])

        # the total shape defines the final output array shape
        if len(self.cube_shapes[0]) > 0:
            totalshape = self.cube_shapes[0]
        else:
            # FIXME: below is not the right default!
            totalshape = np.array(chunkshape)

        return self._read_tsm_file(filename, seqnr, coldesc, totalshape, chunkshape)

    def _read_tsm_file(self, filename, seqnr, coldesc, totalshape, chunkshape, tsm_index=0, offset=0):

        totalshape = np.asarray(totalshape)
        chunkshape = np.asarray(chunkshape)
        chunksize = prod(chunkshape)

        # Need to expose the following somehow
        target_chunksize = None
        memmap = True

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
                    if prod(chunkoversample) * chunksize > target_chunksize:
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
        from casa_formats_io.casa_dask import CASAArrayWrapper

        img_fn = os.path.join(filename, f'table.f{seqnr}_TSM{tsm_index}')

        wrapper = CASAArrayWrapper(img_fn, totalshape, chunkshape,
                                   chunkoversample=chunkoversample, dtype=dtype,
                                   itemsize=itemsize, memmap=memmap, offset=offset)

        # Convert to a dask array
        import uuid
        from dask.array import from_array
        dask_array = from_array(wrapper, name='CASA Data ' + str(uuid.uuid4()),
                                chunks=chunkshape[::-1], meta=np.array([[[]]], dtype=dtype))

        # Since the chunks may not divide the array exactly, all the chunks put
        # together may be larger than the array, so we need to get rid of any
        # extraneous padding.
        final_slice = tuple([slice(dim) for dim in totalshape[::-1]])

        return dask_array[final_slice]


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

        if len(self.last_row_abs.elements) == 0:
            return VariableShapeArrayList([])

        tsm_indices = np.unique(self.cube_index.elements)

        # Start off by reading each TSM file into a dask array
        dask_arrays = {}
        for itsm in tsm_indices:
            dask_arrays[itsm] = self._read_tsm_file(filename, seqnr, coldesc,
                                                    self.cube_shapes[itsm],
                                                    self.tile_shapes[itsm],
                                                    tsm_index=itsm)

        # Next up, we construct for each of the hypercubes an array of row indices
        # in the final table that each hypercube row corresponds to.

        # Start off by making a master index of all rows
        index = da.arange(self.last_row_abs.elements[-1] + 1)

        # Construct the index for each TSM cube
        tsm_row_index = defaultdict(list)
        for idx in range(len(self.cube_index.elements)):
            start = 0 if idx == 0 else self.last_row_abs.elements[idx - 1] + 1
            end = self.last_row_abs.elements[idx] + 1
            tsm_row_index[self.cube_index.elements[idx]].append(index[start:end])

        # Make each row index into a single dask array
        for itsm in tsm_row_index:
            tsm_row_index[itsm] = da.hstack(tsm_row_index[itsm])

        # Return a list of (row_index, hypercube) tuples
        return VariableShapeArrayList([(tsm_row_index[itsm], dask_arrays[itsm]) for itsm in tsm_indices])


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
