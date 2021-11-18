# For convenience, we import the astropy Table class here which
# allows one to do:
#     >>> from casa_formats_io import Table
# instead of:
#     >>> import casa_formats_io
#     >>> from astropy.table import Table
from astropy.table import Table  # noqa

from .casa_dask import *  # noqa
from .casa_low_level_io import *  # noqa
from .casa_wcs import *  # noqa
from .version import version as __version__
from . import table_reader
