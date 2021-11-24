import os
from astropy.table import Table
from astropy.io import registry

from casa_formats_io.casa_low_level_io.table import CASATable


def identify_casa_table(origin, *args, **kwargs):
    if (isinstance(args[2], str) and
            os.path.isdir(args[2]) and
            os.path.exists(os.path.join(args[2], 'table.dat'))):
        with open(os.path.join(args[2], 'table.dat'), 'rb') as f:
            return f.read(4) == b'\xbe\xbe\xbe\xbe'


def read_casa_table(filename, data_desc_id=None):
    table = CASATable.read(filename)
    return table.as_astropy_table(data_desc_id=data_desc_id)


registry.register_identifier('casa-table', Table, identify_casa_table)
registry.register_reader('casa-table', Table, read_casa_table)
