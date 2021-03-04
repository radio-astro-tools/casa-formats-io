0.2 (unreleased)
----------------

- Reduce default target_chunksize to 1000000. [#11]

- Fix bug that occurred when target_chunksize was smaller than the native
  CASA chunk size. [#10]

0.1 (2020-11-08)
----------------

- Initial version which includes ``image_to_dask``, ``getdesc``, ``getdminfo``,
  and ``coordsys_to_astropy_wcs``.
