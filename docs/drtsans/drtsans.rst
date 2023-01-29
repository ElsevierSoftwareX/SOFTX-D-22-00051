drtsans package
===============

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   drtsans.files
   drtsans.mono
   drtsans.plots
   drtsans.tof

Submodules
----------

drtsans.absolute\_units module
------------------------------

.. automodule:: drtsans.absolute_units
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.api module
------------------

.. automodule:: drtsans.api
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.auto\_wedge module
--------------------------

.. automodule:: drtsans.auto_wedge
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.beam\_finder module
---------------------------

.. automodule:: drtsans.beam_finder
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.chopper module
----------------------

.. automodule:: drtsans.chopper
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.dark\_current module
----------------------------

.. automodule:: drtsans.dark_current
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.dataobjects module
--------------------------

This module contains data objects for storing the various projections of I(Q).

.. automodule:: drtsans.dataobjects
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.detector module
-----------------------

.. automodule:: drtsans.detector
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.determine\_bins module
------------------------------

.. automodule:: drtsans.determine_bins
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.frame\_mode module
--------------------------

.. automodule:: drtsans.frame_mode
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.geometry module
-----------------------

.. automodule:: drtsans.geometry
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.instruments module
--------------------------

.. automodule:: drtsans.instruments
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.integrate\_roi module
-----------------------------

.. automodule:: drtsans.integrate_roi
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.iq module
-----------------

.. automodule:: drtsans.iq
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.load module
-------------------

.. automodule:: drtsans.load
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.mask\_utils module
--------------------------

.. automodule:: drtsans.mask_utils
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.meta\_data module
-------------------------

.. automodule:: drtsans.meta_data
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.momentum\_transfer module
---------------------------------

.. automodule:: drtsans.momentum_transfer
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.path module
-------------------

.. automodule:: drtsans.path
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.pixel\_calibration module
---------------------------------

This module contains functions and classes to calculate and apply barscan and tube-widths calibrations.

.. automodule:: drtsans.pixel_calibration
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.prepare\_sensivities\_correction module
-----------------------------------------------

.. automodule:: drtsans.prepare_sensivities_correction
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.process\_uncertainties module
-------------------------------------

.. automodule:: drtsans.process_uncertainties
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.redparms module
-----------------------

.. automodule:: drtsans.redparms
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.reductionlog module
---------------------------

.. automodule:: drtsans.reductionlog
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.resolution module
-------------------------

Smearing Pixels
~~~~~~~~~~~~~~~

There are three different sources for smearing pixel width (X) and height (Y), ranked by their priority:

1. Reduction parameters `smearingPixelSizeX` and `smearingPixelSizeY`,
2. Barscan and tube-width pixel calibrations, and
3. Instrument definition file

A variety of scenarios giving rise to different **final smearing pixel sizes** are presented:

1. If no reduction parameters and no pixel calibration is supplied, then the instrument definition file
provides a smearing width :math:`w_0` and height :math:`h_0` for all pixels.

2. If reduction parameters `smearingPixelSizeX` and `smearingPixelSizeY` are supplied, and no
pixel calibration is supplied, then `smearingPixelSizeX` and `smearingPixelSizeY` become
the smearing width and height for all pixels.

3. If no reduction parameters are supplied but pixel calibration is supplied, then the
smearing width and height are taken from pixel calibration pixel sizes.

4. Finally, if reduction parameters `smearingPixelSizeX` and `smearingPixelSizeY` are supplied,
and a pixel calibration is also supplied, the smearing width :math:`w_i` of pixel :math:`i`
becomes

.. math::

    w_i = \frac{smearingPixelSizeX}{w_0} \cdot w_{pc, i},

where :math:`w_{pc, i}` is the pixel width of pixel :math:`i` provided by the pixel calibration.
An analogous relation follows for the final smearing height.

.. graphviz::

   digraph foo {
      A1 [label="drtsans.mono.biosans.api.load_all_files", shape=box, href="#drtsans.mono.biosans.api.load_all_files"]
      A2 [label="drtsans.mono.gpsans.api.load_all_files", shape=box, href="#drtsans.mono.gpsans.api.load_all_files"]
      A3 [label="drtsans.tof.eqsans.api.load_all_files", shape=box, href="#drtsans.tof.eqsans.api.load_all_files"]
      B1 [label="drtsans.mono.meta_data.set_meta_data", shape=box, href="#drtsans.mono.meta_data.set_meta_data"]
      B2 [label="drtsans.tof.eqsans.meta_data.set_meta_data", shape=box, href="#drtsans.tof.eqsans.meta_data.set_meta_data"]
      C [label="drtsans.geometry.logged_smearing_pixel_size", shape=box, href="#drtsans.geometry.logged_smearing_pixel_size"]
      D1 [label="drtsans.mono.momentum_transfer.retrieve_instrument_setup", shape=box, href="#drtsans.mono.momentum_transfer.retrieve_instrument_setup"]
      D2 [label="drtsans.tof.eqsans.momentum_transfer.retrieve_instrument_setup", shape=box, href="#drtsans.tof.eqsans.momentum_transfer.retrieve_instrument_setup"]
      E [label="drtsans.resolution.InstrumentSetUpParameters", shape=box, href="#drtsans.resolution.InstrumentSetupParameters"]
      F [label="drtsans.resolution.calculate_sigma_theta_geometry", shape=box, fontcolor=blue, href="#drtsans.resolution.calculate_sigma_theta_geometry"]
      A1 -> B1
      A2 -> B1
      A3 -> B2
      B1 -> C
      B2 -> C
      C -> D1 -> E
      C -> D2 -> E
      E -> F;
   }

Above is a **diagram** of the functions involved in porting input reduction parameters `smearingPixelSizeX`
and `smearingPixelSizeY` into the function calculating the undeterminacy in momentum transfer.

.. automodule:: drtsans.resolution
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.samplelogs module
-------------------------

.. automodule:: drtsans.samplelogs
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.save\_2d module
-----------------------

.. automodule:: drtsans.save_2d
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.save\_ascii module
--------------------------

.. automodule:: drtsans.save_ascii
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.script\_utility module
------------------------------

.. automodule:: drtsans.script_utility
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.sensitivity module
--------------------------

.. automodule:: drtsans.sensitivity
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.sensitivity\_correction\_moving\_detectors module
---------------------------------------------------------

.. automodule:: drtsans.sensitivity_correction_moving_detectors
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.sensitivity\_correction\_patch module
---------------------------------------------

.. automodule:: drtsans.sensitivity_correction_patch
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.settings module
-----------------------

.. automodule:: drtsans.settings
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.solid\_angle module
---------------------------

.. automodule:: drtsans.solid_angle
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.stitch module
---------------------

.. automodule:: drtsans.stitch
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.thickness\_normalization module
---------------------------------------

.. automodule:: drtsans.thickness_normalization
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.transmission module
---------------------------

.. automodule:: drtsans.transmission
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.tubecollection module
-----------------------------

.. automodule:: drtsans.tubecollection
   :members:
   :undoc-members:
   :show-inheritance:

drtsans.wavelength module
-------------------------

Disk choppers allow neutrons through with only certain wavelengths. These permitted wavelengths are grouped
into one or more *wavelength bands*. This module provides objects to represent one or a set of wavelength bands.
In addition, the module contains helper functions for conversion between wavelength and time-of-flight, taking
into account the distance traveled by the neutron and its delayed emission time from the moderator.

.. automodule:: drtsans.wavelength
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: drtsans
   :members:
   :undoc-members:
   :show-inheritance:
