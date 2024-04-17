from math import prod

import numpy as np

from .table import CASATable
from .data_managers import StandardStMan, TiledCellStMan

__all__ = ['getdminfo', 'getdesc']


def getdminfo(filename, endian='>'):
    """
    Return the same output as CASA's getdminfo() function, namely a dictionary
    with metadata about the .image file, parsed from the ``table.f0`` file.
    """

    table = CASATable.read(filename, endian=endian)

    colset = table.column_set
    keys = list(colset.data_managers.keys())
    dm = colset.data_managers[keys[0]]

    dminfo = {}

    if isinstance(dm, StandardStMan):

        dminfo['COLUMNS'] = np.array(sorted(col.name for col in colset.columns), 'U')
        dminfo['NAME'] = dm.name
        dminfo['SEQNR'] = 0
        dminfo['TYPE'] = 'StandardStMan'

    dminfo['SPEC'] = {}

    if isinstance(dm, StandardStMan):

        dminfo['SPEC']['BUCKETSIZE'] = dm.bucket_size
        dminfo['SPEC']['IndexLength'] = dm.index_length
        dminfo['SPEC']['MaxCacheSize'] = dm.persistent_cache  # NOTE: not sure if correct
        dminfo['SPEC']['PERSCACHESIZE'] = dm.persistent_cache

    elif isinstance(dm, TiledCellStMan):

        dminfo['SPEC']['DEFAULTTILESHAPE'] = dm.default_tile_shape
        dminfo['SEQNR'] = dm.seqnr
        dminfo['SPEC']['SEQNR'] = dm.seqnr

        dminfo['COLUMNS'] = np.array([dm.column_name], dtype='<U16')
        dminfo['NAME'] = dm.column_name

        dminfo['SPEC']['MAXIMUMCACHESIZE'] = dm.max_cache_size
        dminfo['SPEC']['MaxCacheSize'] = dm.max_cache_size

        bucket = dminfo['SPEC']['HYPERCUBES'] = {}
        bucket = dminfo['SPEC']['HYPERCUBES']['*1'] = {}

        bucket['CubeShape'] = bucket['CellShape'] = dm.cube_shapes[0]
        bucket['TileShape'] = dm.tile_shapes[0]
        bucket['ID'] = {}
        bucket['BucketSize'] = int(dm.total_cube_size /
                                   prod(np.ceil(bucket['CubeShape']
                                              / bucket['TileShape'])))

        dminfo['TYPE'] = 'TiledCellStMan'

    return {'*1': dminfo}


def getdesc(filename, endian='>'):
    """
    Return the same output as CASA's getdesc() function, namely a dictionary
    with metadata about the .image file, parsed from the ``table.dat`` file.
    """

    table = CASATable.read(filename, endian=endian)

    coldesc = table.desc.column_description

    dmkey0 = list(table.column_set.data_managers.keys())[0]

    desc = {}
    for column in coldesc:
        desc[column.name] = {'comment': column.comment,
                             'dataManagerGroup': table.column_set.data_managers[dmkey0].name,
                             'dataManagerType': column.data_manager_type,
                             'keywords': column.keywords.as_dict(),
                             'maxlen': column.maxlen,
                             'option': column.option,
                             'valueType': column.value_type}
    desc['_keywords_'] = table.desc.keywords.as_dict()
    desc['_private_keywords_'] = table.desc.private_keywords.as_dict()
    desc['_define_hypercolumn_'] = {}

    return desc
