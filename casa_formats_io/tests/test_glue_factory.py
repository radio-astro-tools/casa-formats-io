import os

from glue.main import load_plugins
from glue.core.data_factories import load_data


DATA = os.path.join(os.path.dirname(__file__), '..', 'casa_low_level_io',
                    'tests', 'data')


def test_simple():
    load_plugins()
    data = load_data(os.path.join(DATA, 'simple.ms'))
    assert len(data) == 2
    assert data[0].shape == (10, 2)
    assert data[1].shape == (10, 4)
