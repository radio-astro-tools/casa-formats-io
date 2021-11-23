import os

import numpy as np
import dask.array as da

from glue.core import Data
from glue.config import data_factory

from astropy.table import Table

from casa_formats_io.casa_low_level_io.table import CASATable

__all__ = ['read_spectral_cube', 'parse_spectral_cube']


POLARIZATIONS = ['I', 'Q', 'U', 'V',
                 'RR', 'RL', 'LR', 'LL',
                 'XX', 'XY', 'YX', 'YY']


def is_casa_table(filename):
    if (isinstance(filename, str) and
            os.path.isdir(filename) and
            os.path.exists(os.path.join(filename, 'table.dat'))):
        with open(os.path.join(filename, 'table.dat'), 'rb') as f:
            return f.read(4) == b'\xbe\xbe\xbe\xbe'


def table_to_glue_data(table, label):

    # Glue can actually understand astropy tables, but doesn't work well with
    # some columns being vector columns, nor complex numbers, so we expand
    # some of the columns here.

    # Split out vector columns into single columns
    for colname in table.colnames:
        # Make sure we don't accidentally expand big cubes!
        if table[colname].ndim > 1:
            if np.product(table[colname].shape[1:]) > 100:
                raise ValueError(f"Table {colname} is too wide to expand into single columns: {table[colname].shape}")
        if table[colname].ndim == 2:
            for i in range(table[colname].shape[1]):
                table[f'{colname}[{i}]'] = table[colname][:, i]
            table.remove_column(colname)
        elif table[colname].ndim == 3:
            for i in range(table[colname].shape[1]):
                for j in range(table[colname].shape[2]):
                    table[f'{colname}[{i},{j}]'] = table[colname][:, i, j]
            table.remove_column(colname)

    # Split out complex columns into amp/phase/real/imag
    for colname in table.colnames:
        if table[colname].dtype.kind == 'c':
            table[colname + '.amp'] = np.abs(table[colname])
            table[colname + '.phase'] = da.angle(table[colname])
            table[colname + '.real'] = da.real(table[colname])
            table[colname + '.imag'] = da.imag(table[colname])
            table.remove_column(colname)

    kwargs = dict((c, table[c]) for c in table.colnames)

    data = Data(label=label, **kwargs)

    return data


@data_factory(label='CASA Measurement Set', identifier=is_casa_table)
def read_casa_table(filename, **kwargs):

    casa_table = CASATable.read(filename)

    datasets = []
    label_prefix = os.path.basename(filename)

    if casa_table.has_data_desc_id():

        # Extract just the DATA_DESC_ID which can be done as a single table if we
        # get just that column
        data_desc_ids = np.sort(np.unique(casa_table.as_astropy_table(include_columns=['DATA_DESC_ID'])['DATA_DESC_ID']))

        # Load in polarization and spectral window tables
        polarization = Table.read(os.path.join(filename, 'POLARIZATION'), format='casa-table')
        spectral_window = Table.read(os.path.join(filename, 'SPECTRAL_WINDOW'), format='casa-table')

        for data_desc_id in data_desc_ids:

            table = casa_table.as_astropy_table(data_desc_id=data_desc_id)
            datasets.append(table_to_glue_data(table, label=label_prefix + f' [DATA_DESC_ID={data_desc_id}]'))

    else:

        table = casa_table.as_astropy_table()
        datasets.append(table_to_glue_data(table, label=label_prefix))

    return datasets


def setup():
    # This function doesn't need to do anything - it just needs to be present
    # but when this file is loaded the data factory is automatically registered
    pass
