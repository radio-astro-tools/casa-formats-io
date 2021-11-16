from astropy.table import Table
from astropy.io import registry

from casa_formats_io.casa_low_level_io.table import Table as CasaTable


def read_casa_table(filename, data_desc_id=None):
    table = CasaTable.read(filename)
    return table.as_astropy_table(data_desc_id=data_desc_id)


registry.register_reader('casa-table', Table, read_casa_table)
