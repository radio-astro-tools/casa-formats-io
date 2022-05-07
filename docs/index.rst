casa-formats-io documentation
=============================

Scope
-----

The **casa-formats-io** package is a small package which implements
functionality to read data stored in CASA formats (such as .image datasets).
This implementation is independent of and does not use `casacore
<https://casacore.github.io/casacore/>`_. The motivation for this package is
to provide:

* Efficient data access via `dask <https://dask.org/>`_ arrays
* Cross-platform data access, supporting Linux, MacOS X and Windows
* Data access with all modern Python versions, from 3.6 to the latest Python version

At this time (November 2020), only reading .image datasets is supported. Reading measurement sets
(.ms) or writing data of any kind are not yet supported.

casa-formats-io supports python versions >=3.8.

Using casa-formats-io
---------------------

To construct a dask array backed by a .image dataset, use the
:func:`~casa_io_formats.image_to_dask` function::

    >>> from casa_formats_io.casa_dask import image_to_dask
    >>> dask_array = image_to_dask('my_dataset.image/')
    dask.array<CASA Data 6bd6f684-0d21-4614-b953, shape=(2114, 1, 2450, 2450), dtype=float32, chunksize=(14, 1, 350, 2450), chunktype=numpy.ndarray>

Note that rather than use the native CASA chunk size as the size of dask chunks,
which is extremely inefficient for large datasets (for which there may be a
million CASA chunks or more), the :func:`casa_io_formats.image_to_dask` function will
automatically join neighbouring chunks together on-the-fly which then provides
significantly better performance.

In addition to :func:`~casa_io_formats.image_to_dask`, this package
implements :func:`~casa_formats_io.getdesc` and :func:`~casa_formats_io.getdminfo`
which aim to return the same results as CASA's
`getdesc <https://casadocs.readthedocs.io/en/stable/api/tt/stubs.tools.table.html#stubs.tools.table.getdesc>`__
and
`getdminfo <https://casadocs.readthedocs.io/en/stable/api/tt/stubs.tools.table.html#stubs.tools.table.getdminfo>`__
respectively.

Finally, this package provides :func:`~casa_formats_io.coordsys_to_astropy_wcs`) which can
be used to convert CASA WCS information to :class:`~astropy.wcs.WCS` objects.

Table reader (experimental)
---------------------------

This package includes an experimental generic table reader which integrates with
the astropy :class:`~astropy.table.Table` class. To use it, first import
the ``casa_formats_io`` module, which registers the reader, then use the
:meth:`Table.read <astropy.table.Table.read>` method::

    >>> import casa_formats_io
    >>> from astropy.table import Table
    >>> table = Table.read('my_dataset.ms')

If the table contains a ``DATA_DESC_ID`` column, which is the case for e.g.
measurement sets, you will need to also specify the ``data_desc_id=`` argument
to :meth:`Table.read <astropy.table.Table.read>` with a valid integer
DATA_DESC_ID value.

    >>> table_3 = Table.read('my_multims.ms', data_desc_id=3)

Reference/API
-------------

.. automodapi:: casa_formats_io
   :no-inheritance-diagram:
   :inherited-members:
