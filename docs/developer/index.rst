=======================
Developer documentation
=======================

See `solid_angle_correction.py <drtsans/solid_angle_correction.py>`_ for propely documented code

.. _devdocs-standardnames:

--------------------------------------------
Standard variable names and what they denote
--------------------------------------------
* ``wavelength`` (was ``wl``) - wavelength
* ``sample_det_cent`` (was ``s2p``) - sample to detector center distance
* ``l1`` - source to sample distance
* **need definitions of** ``l2``, ``r1``, ``r2``, ``dwl``, ... These should likely get different names with unambigious meanings
* see e.g. solid_angle_correction.py for properly documented/named code


--------------------------------------
Random points about coding conventions
--------------------------------------

* All code will follow the `pep8 standard <https://www.python.org/dev/peps/pep-0008/>`_
* Docstrings will use `numpy formatting <https://numpydoc.readthedocs.io/en/latest/format.html>`_
* Code that uses Mantid algorithms will have an addional "Mantid algorithms used" section with links using inter-sphinx
* All internal functions should have an underbar, as is python standard
* Need more comments in meat of code explaining intent. For example, more clearly named variables (or leave the variables the same but give them better documentation)
* The development team still hasn't decided a standard for how errors/exceptions are handled
