# Numpy + Dask implementation of CASA image data access

from __future__ import print_function, absolute_import, division

import os
from math import ceil, floor
import uuid
import numpy as np

import dask.array

from .casa_low_level_io import getdminfo
from ._casa_chunking import _combine_chunks

__all__ = ['image_to_dask']


def combine_chunks(array_1d, shape, oversample):
    """
    Given a 1d array of values from a CASA file, which contains ``oversample``
    chunks (which is a list/tuple of three elements) return a 1d array that
    would have been returned if there was just one chunk.

    For now, assume that oversample is a tuple of elements in opposite of Numpy
    order for axes.
    """

    # NOTE: if this ends up being a bottleneck, it could also be written as a
    # C/Cython function.

    size = int(np.product(shape))
    native_shape = [s // o for (s, o) in zip(shape, oversample)]
    native_size = int(np.product(native_shape))

    # Split 1-d array into native chunks and add to final array
    result = np.zeros(shape)
    for i in range(oversample[0]):
        for j in range(oversample[1]):
            for k in range(oversample[2]):
                array_current, array_1d = array_1d[:native_size], array_1d[native_size:]
                result[i * native_shape[0]:(i+1) * native_shape[0],
                       j * native_shape[1]:(j+1) * native_shape[1],
                       k * native_shape[2]:(k+1) * native_shape[2]] = array_current.reshape(native_shape, order='F')

    return result.reshape((size,), order='F')


def combine_chunks_c(array_1d, itemsize, shape, oversample):
    if len(shape) == 3:
        shape = tuple(shape) + (1,)
    if len(oversample) == 3:
        oversample = tuple(oversample) + (1,)
    native_shape = [s // o for (s, o) in zip(shape, oversample)]
    return _combine_chunks(np.ascontiguousarray(array_1d), itemsize, *native_shape[::-1], *oversample[::-1])


class CASAArrayWrapper:
    """
    A wrapper class for dask that accesses chunks from a CASA file on request.
    It is assumed that this wrapper will be used to construct a dask array that
    has chunks aligned with the CASA file chunks.

    Having a single wrapper object such as this is far more efficient than
    having one array wrapper per chunk. This is because the dask graph gets
    very large if we end up with one dask array per chunk and slows everything
    down.
    """

    def __init__(self, filename, totalshape, chunkshape, chunkoversample=None, dtype=None, itemsize=None, memmap=False):
        self._filename = filename
        self._totalshape = totalshape[::-1]
        self._chunkshape = chunkshape[::-1]
        self._chunkoversample = chunkoversample[::-1]
        self.shape = totalshape[::-1]
        self.dtype = dtype
        self.ndim = len(self.shape)
        self._stacks = np.ceil(np.array(totalshape) / np.array(chunkshape)).astype(int)
        self._chunksize = np.product(chunkshape)
        self._itemsize = itemsize
        self._memmap = memmap
        if not memmap:
            if self._itemsize == 1:
                self._array = np.unpackbits(np.fromfile(filename, dtype='uint8'), bitorder='little')
            else:
                self._array = np.fromfile(filename, dtype=np.uint8)

    def __getitem__(self, item):

        # TODO: potentially normalize item, for now assume it is a list of slice objects

        indices = []
        for dim in range(self.ndim):
            if isinstance(item[dim], slice):
                indices.append(item[dim].start // self._chunkshape[dim])
            else:
                indices.append(item[dim] // self._chunkshape[dim])

        chunk_number = indices[0]
        for dim in range(1, self.ndim):
            chunk_number = chunk_number * self._stacks[::-1][dim] + indices[dim]

        offset = chunk_number * self._chunksize * self._itemsize

        item_in_chunk = []
        for dim in range(self.ndim):
            if isinstance(item[dim], slice):
                item_in_chunk.append(slice(item[dim].start - indices[dim] * self._chunkshape[dim],
                                      item[dim].stop - indices[dim] * self._chunkshape[dim],
                                      item[dim].step))
            else:
                item_in_chunk.append(item[dim] - indices[dim] * self._chunkshape[dim])
        item_in_chunk = tuple(item_in_chunk)

        if self._itemsize == 1:

            if self._memmap:
                n_native = np.product(self._chunkoversample)
                rounded_up_chunksize = ceil(self._chunksize / n_native / 8) * n_native
                offset = offset // self._chunksize * rounded_up_chunksize
                array_uint8 = np.fromfile(self._filename, dtype=np.uint8,
                                          offset=offset, count=rounded_up_chunksize)
                array_bits = np.unpackbits(array_uint8, bitorder='little')
            else:
                array_bits = self._array[offset * 8: (offset + self._chunksize) * 8]

            chunk = combine_chunks_c(array_bits, 1,
                                            shape=self._chunkshape,
                                            oversample=self._chunkoversample)[:self._chunksize]

            return chunk.reshape(self._chunkshape[::-1], order='F').T[item_in_chunk].astype(np.bool_)

        else:

            if self._memmap:
                data_bytes = np.fromfile(self._filename, dtype=np.uint8,
                                         offset=offset,
                                         count=self._chunksize * self._itemsize)
            else:
                data_bytes = self._array[chunk_number*self._chunksize * self._itemsize:(chunk_number+1)*self._chunksize * self._itemsize]

            return (combine_chunks_c(data_bytes,
                                        self._itemsize,
                                        shape=self._chunkshape,
                                        oversample=self._chunkoversample)
                                    .view(self.dtype)
                                    .reshape(self._chunkshape[::-1], order='F').T[item_in_chunk])


def from_array_fast(arrays, asarray=False, lock=False):
    """
    This is a more efficient alternative to doing::

        [dask.array.from_array(array) for array in arrays]

    that avoids a lot of the overhead in from_array by using the Array
    initializer directly.
    """
    slices = tuple(slice(0, size) for size in arrays[0].shape)
    chunk = tuple((size,) for size in arrays[0].shape)
    meta = np.zeros((0,), dtype=arrays[0].dtype)
    dask_arrays = []
    for array in arrays:
        name1 = str(uuid.uuid4())
        name2 = str(uuid.uuid4())
        dsk = {(name1,) + (0,) * array.ndim: (dask.array.core.getter, name2,
                                              slices, asarray, lock),
               name2: array}
        dask_arrays.append(dask.array.Array(dsk, name1, chunk, meta=meta, dtype=array.dtype))
    return dask_arrays


def image_to_dask(imagename, memmap=True, mask=False, target_chunksize=None):
    """
    Read a CASA image (a folder containing a ``table.f0_TSM0`` file) into a
    numpy array.

    Parameters
    ----------
    imagename : str
        The filename of the CASA image directory
    memmap : bool, optional
        Whether to use memory mapping or load the full cube into memory
    mask : str or bool, optional
        If set to a string, should be the name of the mask (e.g. ``mask1``) and
        the mask is returned instead of the data. If `True`, ``mask0`` will be used.
    target_chunksize : int
        The desired dask chunk size - CASA chunks are aggregated into blocks of
        approximately this number of elements.
    """

    # the data is stored in the following binary file
    # each of the chunks is stored on disk in fortran-order
    if mask:
        if mask is True:
            mask = 'mask0'
        imagename = os.path.join(str(imagename), mask)

    if not os.path.exists(imagename):
        raise FileNotFoundError(imagename)

    # the data is stored in the following binary file
    # each of the chunks is stored on disk in fortran-order
    img_fn = os.path.join(str(imagename), 'table.f0_TSM0')

    # load the metadata from the image table. Note that this uses our own
    # implementation of getdminfo, which is equivalent to
    # from casatools import table
    # tb = table()
    # tb.open(str(imagename))
    # dminfo = tb.getdminfo()
    # tb.close()
    dminfo = getdminfo(str(imagename))

    # Determine whether file is big endian
    big_endian = dminfo['*1']['BIGENDIAN']

    # chunkshape defines how the chunks (array subsets) are written to disk
    chunkshape = tuple(dminfo['*1']['SPEC']['DEFAULTTILESHAPE'])
    chunksize = np.product(chunkshape)

    # the total shape defines the final output array shape
    totalshape = dminfo['*1']['SPEC']['HYPERCUBES']['*1']['CubeShape']

    # we expect that the total size of the array will be determined by finding
    # the number of chunks along each dimension rounded up
    totalsize = np.product(np.ceil(totalshape / chunkshape)) * chunksize

    # the file size helps us figure out what the dtype of the array is
    filesize = os.stat(img_fn).st_size

    # the ratio between these tells you how many chunks must be combined
    # to create a final stack
    stacks = np.ceil(totalshape / chunkshape).astype(int)
    nchunks = int(np.product(stacks))

    # check that the file size is as expected and determine the data dtype
    if mask:
        expected = nchunks * ceil(chunksize / 8)
        if filesize != expected:
            raise ValueError("Unexpected file size for mask, found {0} but "
                             "expected {1}".format(filesize, expected))
        dtype = bool
        itemsize = 1
    else:
        if filesize == totalsize * 4:
            if big_endian:
                dtype = '>f4'
            else:
                dtype = '<f4'
            itemsize = 4
        elif filesize == totalsize * 8:
            if big_endian:
                dtype = '>f8'
            else:
                dtype = '<f8'
            itemsize = 8
        else:
            raise ValueError("Unexpected file size for data, found {0} but "
                             "expected {1} or {2}".format(filesize, totalsize * 4, totalsize * 8))

    # CASA does not like numpy ints!
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
                previous_chunkoversample = chunkoversample
            if finished:
                break

    chunkshape = [c * o for (c, o) in zip(chunkshape, chunkoversample)]

    # Create a wrapper that takes slices and returns the appropriate CASA data
    wrapper = CASAArrayWrapper(img_fn, totalshape, chunkshape, chunkoversample=chunkoversample, dtype=dtype, itemsize=itemsize, memmap=memmap)

    # Convert to a dask array
    dask_array = dask.array.from_array(wrapper, name='CASA Data ' + str(uuid.uuid4()), chunks=chunkshape[::-1])

    # Since the chunks may not divide the array exactly, all the chunks put
    # together may be larger than the array, so we need to get rid of any
    # extraneous padding.
    final_slice = tuple([slice(dim) for dim in totalshape[::-1]])

    return dask_array[final_slice]
