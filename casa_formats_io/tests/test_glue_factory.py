import os
import pytest

DATA = os.path.join(os.path.dirname(__file__), '..', 'casa_low_level_io',
                    'tests', 'data')


def test_simple_direct_factory():
    pytest.importorskip('glue')
    # This import is deliberately here to not automatically register
    # the plugin for the whole test file - we want to make sure
    # in test_simple_load_data that the plugin registration works.
    from casa_formats_io.glue_factory import read_casa_table
    data = read_casa_table(os.path.join(DATA, 'simple.ms'))
    assert len(data) == 2
    assert data[0].shape == (10, 2)
    assert data[1].shape == (10, 4)


def test_simple_load_data():
    pytest.importorskip('glue')
    from glue.main import load_plugins
    from glue.core.data_factories import load_data
    load_plugins()
    data = load_data(os.path.join(DATA, 'simple.ms'))
    assert len(data) == 2
    assert data[0].shape == (10, 2)
    assert data[1].shape == (10, 4)
