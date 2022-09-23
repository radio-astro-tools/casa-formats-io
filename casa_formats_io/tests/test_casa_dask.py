from __future__ import print_function, absolute_import, division

import os
from itertools import product

import pytest
import numpy as np
from numpy.testing import assert_allclose

from ..casa_dask import image_to_dask

try:
    from casatools import image
    CASA_INSTALLED = True
except ImportError:
    try:
        from taskinit import ia as image
        CASA_INSTALLED = True
    except ImportError:
        CASA_INSTALLED = False

# NOTE: the (127, 337, 109) example is to make sure that things work correctly
# when the shape isn't a multiple of the chunk size along any
# dimension.
SHAPES = [(3, 4, 5), (129, 128, 130), (513, 128, 128), (128, 513, 128),
          (128, 128, 513), (512, 64, 64), (127, 337, 109)]


@pytest.mark.skipif(not CASA_INSTALLED, reason='CASA tests must be run in a CASA environment.')
@pytest.mark.parametrize(('memmap', 'shape'), product([False, True], SHAPES))
def test_image_to_dask(tmpdir, memmap, shape):

    # Unit tests for the low-level image_to_dask function which can
    # read a CASA image or mask to a Dask array.

    reference = np.random.random(shape).astype(np.float32)

    # CASA seems to have precision issues when computing masks with values
    # very close to e.g. 0.5 in >0.5. To avoid this, we filter out random
    # values close to the boundaries that we use below.
    reference[np.isclose(reference, 0.2)] += 0.05
    reference[np.isclose(reference, 0.5)] += 0.05
    reference[np.isclose(reference, 0.8)] += 0.05

    os.chdir(tmpdir.strpath)

    # Start off with a simple example with no mask. Note that CASA requires
    # the array to be transposed in order to match what we would expect.

    ia = image()
    ia.fromarray('basic.image', pixels=reference.T, log=False)
    ia.close()

    array1 = image_to_dask('basic.image', memmap=memmap)
    assert array1.dtype == np.float32
    assert_allclose(array1, reference)

    # Check slicing
    assert_allclose(array1[:2, :1, :3], reference[:2, :1, :3])

    # Try and get a mask - this should fail since there isn't one.

    with pytest.raises(FileNotFoundError):
        image_to_dask('basic.image', mask=True, memmap=memmap)

    # Now create an array with a simple uniform mask.

    ia = image()
    ia.fromarray('scalar_mask.image', pixels=reference.T, log=False)
    ia.calcmask(mask='T')
    ia.close()

    array2 = image_to_dask('scalar_mask.image', memmap=memmap)
    assert_allclose(array2, reference)

    mask2 = image_to_dask('scalar_mask.image', mask=True, memmap=memmap)
    assert mask2.dtype is np.dtype('bool')
    assert mask2.shape == array2.shape
    assert np.all(mask2)

    # Check with a full 3-d mask

    ia = image()
    ia.fromarray('array_mask.image', pixels=reference.T, log=False)
    ia.calcmask(mask='array_mask.image>0.5')
    ia.close()

    array3 = image_to_dask('array_mask.image', memmap=memmap)
    assert_allclose(array3, reference)

    mask3 = image_to_dask('array_mask.image', mask=True, memmap=memmap)
    assert_allclose(mask3, reference > 0.5)

    # Check slicing
    assert_allclose(mask3[:2, :1, :3], (reference > 0.5)[:2, :1, :3])

    # Test specifying the mask name

    ia = image()
    ia.fromarray('array_masks.image', pixels=reference.T, log=False)
    ia.calcmask(mask='array_masks.image>0.5')
    ia.calcmask(mask='array_masks.image>0.2')
    ia.calcmask(mask='array_masks.image>0.8', name='gt08')
    ia.close()

    array4 = image_to_dask('array_masks.image', memmap=memmap)
    assert_allclose(array4, reference)

    mask4 = image_to_dask('array_masks.image', mask=True, memmap=memmap)
    assert_allclose(mask4, reference > 0.5)

    mask5 = image_to_dask('array_masks.image', mask='mask0', memmap=memmap)
    assert_allclose(mask5, reference > 0.5)

    mask6 = image_to_dask('array_masks.image', mask='mask1', memmap=memmap)
    assert_allclose(mask6, reference > 0.2)

    mask7 = image_to_dask('array_masks.image', mask='gt08', memmap=memmap)
    assert_allclose(mask7, reference > 0.8)

    # Check that things still work if we write the array out with doubles

    reference = np.random.random(shape).astype(np.float64)

    ia = image()
    ia.fromarray('double.image', pixels=reference.T, type='d', log=False)
    ia.close()

    array8 = image_to_dask('double.image', memmap=memmap)
    assert array8.dtype == np.float64
    assert_allclose(array8, reference)


@pytest.mark.skipif(not CASA_INSTALLED, reason='CASA tests must be run in a CASA environment.')
def test_target_chunksize():

    reference = np.random.random((256, 256, 256))

    ia = image()
    ia.fromarray('large.image', pixels=reference, log=False)
    ia.close()

    array1 = image_to_dask('large.image')
    assert array1.chunksize == (32, 128, 256)

    array2 = image_to_dask('large.image', target_chunksize=100000)
    assert array2.chunksize == (32, 32, 64)

    array3 = image_to_dask('large.image', target_chunksize=1000)
    assert array3.chunksize == (32, 32, 32)

    array4 = image_to_dask('large.image', target_chunksize=1000000000)
    assert array4.chunksize == (256, 256, 256)
