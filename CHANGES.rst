0.2.1 (2022-09-29)
------------------

- Remove old casa_low_level_io.py file which should have been removed when
  the casa_formats_io.casa_low_level_io sub-package was created. [#54]

0.2 (2022-09-23)
----------------

- Fix bug that occurred when target_chunksize was smaller than the native
  CASA chunk size. [#10]

- Started implementing a generic CASA table reader. [#12, #14, #15, #22, #24, #26, #30, #32, #35, #38]

- Add a glue data factory for CASA tables. [#33]

0.1 (2020-11-08)
----------------

- Initial version which includes ``image_to_dask``, ``getdesc``, ``getdminfo``,
  and ``coordsys_to_astropy_wcs``.
