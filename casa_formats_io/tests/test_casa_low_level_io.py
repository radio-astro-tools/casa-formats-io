from __future__ import print_function, absolute_import, division

import os
import pytest
import numpy as np
from numpy.testing import assert_equal
from astropy.table import Table
from pprint import pformat

from ..casa_low_level_io import TiledCellStMan, getdminfo, getdesc, EndianAwareFileHandle
# from ...tests.test_casafuncs import make_casa_testimage

try:
    from casatools import table, image
    CASATOOLS_INSTALLED = True
except ImportError:
    CASATOOLS_INSTALLED = False

DATA = os.path.join(os.path.dirname(__file__), 'data')


SHAPES = [(3,), (5, 3), (8, 4, 2), (4, 8, 3, 1), (133, 400), (100, 211, 201),
          (50, 61, 72, 83), (4, 8, 10, 20, 40)]


@pytest.mark.skipif('not CASATOOLS_INSTALLED')
@pytest.mark.parametrize('shape', SHAPES)
def test_getdminfo(tmp_path, shape):

    filename = str(tmp_path / 'test.image')

    data = np.random.random(shape)

    ia = image()
    ia.fromarray(outfile=filename, pixels=data, log=False)
    ia.close()

    tb = table()
    tb.open(filename)
    reference = tb.getdminfo()
    tb.close()

    actual = getdminfo(filename)

    # The easiest way to compare the output is simply to compare the output
    # from pformat (checking for dictionary equality doesn't work because of
    # the Numpy arrays inside). Older versions of casatools represent some of
    # the vectors as int32 instead of native int but our implementation uses
    # native int so strip any mention of int32 from reference output
    assert pformat(actual) == pformat(reference).replace(', dtype=int32', '')


def test_getdminfo_large():

    # Check that things continue to work fine when we cross the threshold from
    # a dataset with a size that can be represented by a 32-bit integer to one
    # where the size requires a 64-bit integer. We use pre-generated
    # table.f0 files here since generating these kinds of datasets is otherwise
    # slow and consumes a lot of memory.

    filename = os.path.join(DATA, 'lt32bit.image')
    with open(os.path.join(filename, 'table.f0'), 'rb') as f_orig:
        f = EndianAwareFileHandle(f_orig, '>', filename)
        magic = f.read(4)
        lt32bit = TiledCellStMan()
        lt32bit.read_header(f)
    assert_equal(lt32bit.cube_shape, (320, 320, 1, 1920))

    filename = os.path.join(DATA, 'gt32bit.image')
    with open(os.path.join(filename, 'table.f0'), 'rb') as f_orig:
        f = EndianAwareFileHandle(f_orig, '>', filename)
        magic = f.read(4)
        gt32bit = TiledCellStMan()
        gt32bit.read_header(f)
    assert_equal(gt32bit.cube_shape, (640, 640, 1, 1920))


@pytest.fixture
def filename(request):
    return request.getfixturevalue(request.param)


@pytest.mark.openfiles_ignore
@pytest.mark.skipif('not CASATOOLS_INSTALLED')
def test_generic_table_read(tmp_path):

    # NOTE: for now, this doesn't check that we can read the data - just
    # the metadata about the table.

    filename_fits = str(tmp_path / 'generic.fits')
    filename_casa = str(tmp_path / 'generic.image')

    N = 120

    t = Table()
    t['short'] = np.arange(N, dtype=np.int16)
    t['ushort'] = np.arange(N, dtype=np.uint16)
    t['int'] = np.arange(N, dtype=np.int32)
    t['uint'] = np.arange(N, dtype=np.uint32)
    t['float'] = np.arange(N, dtype=np.float32)
    t['double'] = np.arange(N, dtype=np.float64)
    t['complex'] = np.array([1 + 2j, 3.3 + 8.2j, -1.2 - 4.2j] * (N // 3), dtype=np.complex64)
    t['dcomplex'] = np.array([3.33 + 4.22j, 3.3 + 8.2j, -1.2 - 4.2j] * (N // 3), dtype=np.complex128)
    t['str'] = np.array(['reading', 'casa', 'images'] * (N // 3))

    # Repeat this at the end to make sure we correctly finished reading
    # the complex column metadata
    t['int2'] = np.arange(N, dtype=np.int32)

    t.write(filename_fits)

    tb = table()
    tb.fromfits(filename_casa, filename_fits)
    tb.close()

    # Use the arrays in the table to also generate keywords of various types
    keywords = {'scalars': {}, 'arrays': {}}
    for name in t.colnames:
        keywords['scalars']['s_' + name] = t[name][0]
        keywords['arrays']['a_' + name] = t[name]

    tb.open(filename_casa)
    tb.putkeywords(keywords)
    tb.flush()
    tb.close()

    desc_actual = getdesc(filename_casa, endian='<')

    tb.open(filename_casa)
    desc_reference = tb.getdesc()
    reference_getdminfo = tb.getdminfo()
    tb.close()

    # Older versions of casatools represent some of the vectors as int32
    # instead of native int but our implementation uses native int so strip any
    # mention of int32 from reference output
    assert pformat(desc_actual) == pformat(desc_reference).replace(', dtype=int32', '')

    actual_getdminfo = getdminfo(filename_casa, endian='<')

    # FIXME: For some reason IndexLength is zero in the CASA output
    actual_getdminfo['*1']['SPEC']['IndexLength'] = 0

    assert pformat(actual_getdminfo) == pformat(reference_getdminfo)


def test_getdesc_floatarray():

    # There doesn't seem to be an easy way to create CASA images
    # with float (not double) arrays. In test_getdesc, all the floats
    # end up getting converted to double. So instead we use a table.dat
    # file that was found in the wild.

    desc = getdesc(os.path.join(DATA, 'floatarray.image'))
    trc = desc['_keywords_']['masks']['mask0']['box']['trc']
    assert trc.dtype == np.float32
    assert_equal(trc, [512, 512, 1, 100])


@pytest.mark.skipif('not CASATOOLS_INSTALLED')
def test_logtable(tmp_path):

    filename = str(tmp_path / 'test.image')
    logtable = str(tmp_path / 'test.image' / 'logtable')

    data = np.random.random((2, 3, 4))

    ia = image()
    ia.fromarray(outfile=filename, pixels=data, log=False)
    ia.sethistory(origin='test', history=['a', 'bb', 'ccccccccccc'] * 23)
    ia.close()

    tb = table()
    tb.open(logtable)
    reference_getdesc = tb.getdesc()
    reference_getdminfo = tb.getdminfo()
    tb.close()

    actual_getdesc = getdesc(logtable, endian='<')
    actual_getdminfo = getdminfo(logtable, endian='<')

    assert pformat(actual_getdesc) == pformat(reference_getdesc)
    assert pformat(actual_getdminfo) == pformat(reference_getdminfo)
