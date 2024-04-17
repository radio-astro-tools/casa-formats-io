from __future__ import print_function, absolute_import, division

import os
import glob
import pytest
import numpy as np
from numpy.testing import assert_equal
from astropy.table import Table
from pprint import pformat

from ..data_managers import TiledCellStMan
from ..casa_functions import getdminfo, getdesc
from ..core import EndianAwareFileHandle
from ..table import CASATable

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
    assert pformat(actual) == pformat(reference).replace(', dtype=int32', '').replace('U3', 'U16')


def test_getdminfo_large():

    # Check that things continue to work fine when we cross the threshold from
    # a dataset with a size that can be represented by a 32-bit integer to one
    # where the size requires a 64-bit integer. We use pre-generated
    # table.f0 files here since generating these kinds of datasets is otherwise
    # slow and consumes a lot of memory.

    filename = os.path.join(DATA, 'lt32bit.image')
    with open(os.path.join(filename, 'table.f0'), 'rb') as f_orig:
        f = EndianAwareFileHandle(f_orig, '>', filename)
        magic = f.read(4)  # noqa
        lt32bit = TiledCellStMan()
        lt32bit.read_header(f)
    assert_equal(lt32bit.cube_shapes[0], (320, 320, 1, 1920))

    filename = os.path.join(DATA, 'gt32bit.image')
    with open(os.path.join(filename, 'table.f0'), 'rb') as f_orig:
        f = EndianAwareFileHandle(f_orig, '>', filename)
        magic = f.read(4)  # noqa
        gt32bit = TiledCellStMan()
        gt32bit.read_header(f)
    assert_equal(gt32bit.cube_shapes[0], (640, 640, 1, 1920))


@pytest.fixture
def filename(request):
    return request.getfixturevalue(request.param)


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

    tb.open(filename_casa, nomodify=False)
    tb.putkeywords(keywords)
    tb.flush()
    tb.close()

    desc_actual = getdesc(filename_casa, endian='<')

    tb.open(filename_casa)
    desc_reference = tb.getdesc()
    reference_getdminfo = tb.getdminfo()
    tb.close()

    assert pformat(desc_actual).replace(', dtype=int32', '') == pformat(desc_reference).replace(', dtype=int32', '')

    actual_getdminfo = getdminfo(filename_casa, endian='<')

    assert pformat(actual_getdminfo).replace('<U16', '<U8') == pformat(reference_getdminfo)

    tnew = Table.read(filename_casa)


def test_getdesc_floatarray():

    # There doesn't seem to be an easy way to create CASA images
    # with float (not double) arrays. In test_getdesc, all the floats
    # end up getting converted to double. So instead we use a table.dat
    # file that was found in the wild.

    desc = getdesc(os.path.join(DATA, 'floatarray.image'))
    trc = desc['_keywords_']['masks']['mask0']['box']['trc']
    assert trc.dtype == np.dtype('<f4')
    assert_equal(trc, [512, 512, 1, 100])


@pytest.mark.skipif('not CASATOOLS_INSTALLED')
def test_logtable(tmp_path):

    filename = str(tmp_path / 'test.image')
    logtable = str(tmp_path / 'test.image' / 'logtable')

    data = np.random.random((2, 3, 4))

    ia = image()
    ia.fromarray(outfile=filename, pixels=data, log=False)
    ia.sethistory(origin='test', history=['a', 'bb', 'ccccccccccc'] * 3247)
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

    Table.read(logtable)


@pytest.mark.parametrize('tablename', ('.',
                                       'ANTENNA',
                                       'CALDEVICE',
                                       'DATA_DESCRIPTION',
                                       'FEED',
                                       'FIELD',
                                       'FLAG_CMD',
                                       'HISTORY',
                                       'OBSERVATION',
                                       'POINTING',
                                       'POLARIZATION',
                                       'PROCESSOR',
                                       'SOURCE',
                                       'SPECTRAL_WINDOW',
                                       'STATE',
                                       'SYSCAL',
                                       'SYSPOWER'))
def test_ms_tables(tablename):

    table_filename = os.path.join(DATA, 'simple.ms', tablename)

    # Concatenation issue as arrays change shape half way through
    if tablename == 'SYSPOWER':
        pytest.xfail()

    tt = [Table.read(table_filename, data_desc_id=0),
          Table.read(table_filename, data_desc_id=1)]

    if CASATOOLS_INSTALLED:

        tb = table()
        tb.open(table_filename)

        for colname in tb.colnames():

            # CASA has issues reading this in
            if tablename == 'SOURCE' and colname in ['POSITION', 'TRANSITION']:
                continue
            if tablename == 'CALDEVICE' and colname in ['CAL_EFF', 'TEMPERATURE_LOAD']:
                continue
            if tablename == 'SPECTRAL_WINDOW' and colname in ['CHAN_FREQ', 'CHAN_WIDTH', 'EFFECTIVE_BW',
                                                              'RESOLUTION', 'ASSOC_SPW_ID', 'ASSOC_NATURE']:
                continue

            # Column is empty? browsetable also has issues reading this properly
            if tablename == '.' and colname in ['FLAG_CATEGORY']:
                continue

            if tablename == '.':
                # Check two main sections of table, which have different shapes for
                # some of the columns
                assert_equal(tt[0][colname], tb.getcol(colname, startrow=0, nrow=10).T)
                assert_equal(tt[1][colname], tb.getcol(colname, startrow=10, nrow=10).T)
            else:
                assert_equal(tt[0][colname], tb.getcol(colname).T)

        tb.close()


@pytest.mark.skipif('not CASATOOLS_INSTALLED')
def test_vector_columns(tmp_path):

    filename_fits = str(tmp_path / 'vector.fits')
    filename_casa = str(tmp_path / 'vector.image')

    N = 120

    t = Table()
    t['short'] = np.arange(N, dtype=np.int16).reshape((5, 4, 3, 2))
    t['int'] = np.arange(N, dtype=np.int32).reshape((5, 4, 3, 2))
    t['double'] = np.arange(N, dtype=np.float64).reshape((5, 4, 3, 2))
    t.write(filename_fits)

    tb = table()
    tb.fromfits(filename_casa, filename_fits)
    tb.close()

    t2 = Table.read(filename_casa)

    assert_equal(t['short'], t2['short'])
    assert_equal(t['int'], t2['int'])
    assert_equal(t['double'], t2['double'])


# Test reading tables in the casadata package

try:
    from casadata import datapath
except ImportError:
    CASADATA_TABLES = []
    CASADATA_INSTALLED = True
else:
    CASADATA_TABLES = [os.path.dirname(x) for x in glob.glob(os.path.join(datapath, '**', 'table.dat'), recursive=True)]
    CASADATA_INSTALLED = False


EXPECTED_FAILURES = [
    'ephemerides/Sources',  # our table is too long
]


@pytest.mark.parametrize('table_filename', CASADATA_TABLES)
def test_casadata(table_filename):

    if os.path.relpath(table_filename, datapath) in EXPECTED_FAILURES:
        pytest.xfail()

    tt = Table.read(table_filename)

    if CASATOOLS_INSTALLED:

        tb = table()
        tb.open(table_filename)

        expected_colnames = tb.colnames()

        # Some tables contain a Spectral_Record column that we can't read yet
        # but that browsetable also can't read/display so we don't support this.
        if 'Spectral_Record' in expected_colnames:
            expected_colnames.remove('Spectral_Record')

        assert tt.colnames == expected_colnames

        for colname in expected_colnames:
            try:
                expected_data = tb.getcol(colname).T
            except Exception:
                # If casa can't read it, we can assume we don't need to support it
                continue
            assert_equal(tt[colname], expected_data)

        tb.close()
