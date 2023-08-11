import os
from math import prod

import numpy as np
import dask.array as da

from glue.core import Data
from glue.config import data_factory
from glue.core.coordinates import AffineCoordinates

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

    # Split out vector columns into single columns unless all vector columns
    # are the same

    reference_shape = table[table.colnames[0]].shape
    for colname in table.colnames[1:]:
        if table[colname].shape != reference_shape:
            split = True
            break
    else:
        split = False

    if split:
        for colname in table.colnames:
            if table[colname].ndim > 1:
                if prod(table[colname].shape[1:]) > 100:
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


def ms_table_to_glue_data(table, label, polarizations):

    # For MS tables specifically we can be smarter about how we split columns
    # as we can special case different columns.

    data_shape = table['DATA'].shape

    # Split UVW vector column into U, V, and W
    table['U'] = table['UVW'][:, 0]
    table['V'] = table['UVW'][:, 1]
    table['W'] = table['UVW'][:, 2]
    table.remove_column('UVW')

    # Columns that have polarization dimensions can be split
    for colname in ['FLAG', 'WEIGHT', 'SIGMA', 'DATA', 'MODEL_DATA', 'CORRECTED_DATA']:
        if colname in table.colnames:
            for ipol, pol in enumerate(polarizations):
                table[colname + '_' + pol] = table[colname][..., ipol]
            table.remove_column(colname)

    # Now we are left with some 1-D and some 2-D columns, so we broadcast all
    # the 1-D ones to match the 2-D ones.
    for colname in table.colnames:
        if table[colname].ndim == 1:
            reshaped_data = da.broadcast_to(table[colname], data_shape[:2][::-1]).T
            table.remove_column(colname)
            table[colname] = reshaped_data

    # Split out complex columns into amp/phase/real/imag
    for colname in table.colnames:
        if table[colname].dtype.kind == 'c':
            table[colname + '.amp'] = np.abs(table[colname])
            table[colname + '.phase'] = da.angle(table[colname])
            table[colname + '.real'] = da.real(table[colname])
            table[colname + '.imag'] = da.imag(table[colname])
            table.remove_column(colname)

    kwargs = dict((c, table[c]) for c in table.colnames)

    # For now just set up an identity transform but with correct axis labels
    # to avoid confusion.
    coords = AffineCoordinates(np.identity(3), labels=['Frequency', 'Row index'])

    data = Data(label=label, coords=coords, **kwargs)

    return data

# We set the priority here to make sure that we rank higher than the default
# astropy Table factory.
@data_factory(label='CASA Table', identifier=is_casa_table, priority=1)
def read_casa_table(filename, **kwargs):

    casa_table = CASATable.read(filename)

    datasets = []
    label_prefix = os.path.basename(filename)

    if casa_table.has_data_desc_id():

        # Extract just the DATA_DESC_ID which can be done as a single table if we
        # get just that column
        data_desc_ids = np.sort(np.unique(casa_table.as_astropy_table(include_columns=['DATA_DESC_ID'])['DATA_DESC_ID']))

        # Load in polarization and spectral window tables
        table_desc = Table.read(os.path.join(filename, 'DATA_DESCRIPTION'), format='casa-table')
        table_pol = Table.read(os.path.join(filename, 'POLARIZATION'), format='casa-table')
        # table_spw = Table.read(os.path.join(filename, 'SPECTRAL_WINDOW'), format='casa-table')

        for data_desc_id in data_desc_ids:

            # Extract polarizations
            pol_id = table_desc['POLARIZATION_ID'][data_desc_id]
            corr_type = table_pol['CORR_TYPE'][pol_id]
            polarizations = [POLARIZATIONS[ct - 1] for ct in corr_type]

            # Extract frequencies
            # spw_id = table_desc['SPECTRAL_WINDOW_ID'][data_desc_id]
            # chan_freq = table_spw['CHAN_FREQ'][spw_id]

            table = casa_table.as_astropy_table(data_desc_id=data_desc_id)
            datasets.append(ms_table_to_glue_data(table, label_prefix + f' [DATA_DESC_ID={data_desc_id}]', polarizations))

    else:

        table = casa_table.as_astropy_table()
        datasets.append(table_to_glue_data(table, label_prefix))

    return datasets


def setup():
    # This function doesn't need to do anything - it just needs to be present
    # but when this file is loaded the data factory is automatically registered
    pass
