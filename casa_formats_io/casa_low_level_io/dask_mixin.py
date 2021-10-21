# The helpers here make it possible to use dask arrays as columns in astropy
# tables.

# https://github.com/astropy/astropy/pull/12219

from astropy.utils.data_info import ParentDtypeInfo

import dask.array as da

__all__ = ['dask_to_mixin']


class DaskInfo(ParentDtypeInfo):
    @staticmethod
    def default_format(val):
        return f'{val.compute()}'


class DaskColumn(da.Array):

    info = DaskInfo()

    def copy(self):
        # Array hard-codes the resulting copied array as Array, so need to
        # overload this since Table tries to copy the array.
        return DaskColumn(self.dask, self.name, self.chunks, meta=self)

    def __getitem__(self, item):
        arr = super().__getitem__(item)
        return DaskColumn(arr.dask, arr.name, arr.chunks, meta=arr)



def dask_to_mixin(arr):
    return DaskColumn(arr.dask, arr.name, arr.chunks, meta=arr)
