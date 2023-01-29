********************
Onboarding Checklist
********************

This check list is to assist developers who are new to the SANS backend software project as the onboarding process.


How to get started
##################


Accounts setup
**************

A new developer, who joins the SANS backend software project, shall need to have or request access to the environments and tools.  There is a contact person for requesting access in each section.

* Project managers

  * John Hetrick
  * Peter F. Peterson

* drtSANS repository

  * URL: https://code.ornl.gov/ns-hfir-scse/sans/sans-backend/
  * You will need permission to commit changes
  * If you don't have access, please contact the project managers(s).

* Slack channels/Onboarding buddy

  * ornlccsd.slack.com # sae-neutrons
 
    Please contact the Project Managers for the access to the slack channel.
    The project managers will know who to send an email to requesting access to our neutron slack channels. 
    There may be more than one and if the person is outside CSMD that is a special invite.

* SNS analysis cluster account and integration data access

  drtSANS is designed for users to process SANS data on SNS analysis cluster or
  on SNS jupyter hub.
  Therefore, drtSANS shall be tested on SNS analysis cluster.

  * Analysis cluster web access URL: https://analysis.sns.gov
  * Jupyter hub access URL: https://jupyter.sns.gov/hub/login
  * Analysis cluter server address: analysis.sns.gov
  * Portions of the test suite requires read and write access to `/SNS/EQSANS/shared/sans-backend` on SNS analysis cluster.
  * If you have any issue with accessiblity, please contact the project managers(s).

* Access to SNS archive data

  SNS and HFIR SANS data files are relatively large.
  Developers are expected to work with the raw data files residing on the SNS/HFIR data archive.
  Therefore it is necessary for drtSANS developers to access `/SNS/EQSANS`, `/HFIR/CG2/` and `/HFIR/CG3`.

  * Access can be required by emailing SNS Linux support (linuxsupport@ornl.gov) and cc-ing neutron project managers and your group leader.

* Status meeting

  * All developers shall attend morning neutron status meeting.
  * Contact project managers to be invited.


Set-up for development in a virtual environment
***********************************************

Refer to section *Set-up for development in a virtual environment* in :doc:`README_developer <README_developer>`.

Building the documentation
##########################

.. _building_docs:

The site can be build directly using

.. code-block:: shell

   $ sphinx-build -b html docs/ build/sphinx/html

or

.. code-block:: shell

   $ python setup.py build_sphinx



Development procedure
#####################

How to develop codes in drtSANS shall follow the instruction in `CONTRIBUTION <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/-/blob/next/CONTRIBUTING.rst>`_.

..
  1. A developer is assigned with a task during neutron status meeting and changes the task's status to **In Progress**.
  2. The developer creates a branch off *next* and completes the task in this branch.
  3. The developer creates a merge request (MR) off *next*.
  4. The developer asks for another developer as a reviewer to review the MR.  An MR can only be approved and merged by the reviewer.
  5. The developer changes the task’s status to **Complete**.


Test Driven Development (TDD)
#############################

* Test driven Development

   drtSANS development follows `test-driven development <https://en.wikipedia.org/wiki/Test-driven_development>`_ (TDD) process [1].
   All software requirements for SANS data reduction shall be converted to test cases before software is fully developed.
   All software developments are tracked by repeatedly testing the software against all test cases.

* Unit test
   
  All methods and modules shall have unit tests implemented.
  Unit tests are located in `repo/tests/unit/new <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/-/tree/next/tests/unit/new>`_.
  A unit test shall be created in the corresponding directory to the method or module that it tests against.

  Examples:

  * `drtsans/resolution.py <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/-/blob/next/drtsans/resolution.py>`_ and `tests/unit/new/drtsans/test_resolution.py <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/-/blob/next/tests/unit/new/drtsans/test_resolution.py>`_.
  * `drtsans/tof/eqsans/incoherence_correction_q1d.py <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/-/blob/next/drtsans/tof/eqsans/incoherence_correction_1d.py>`_ and `tests/unit/new/drtsans/tof/eqsans/test_incoherence_correction_q1d.py <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/-/blob/next/tests/unit/new/drtsans/tof/eqsans/test_incoherence_correction_q1d.py>`_.

* Integration test

  Integration test will test the combination of Individual modules and methods.
  Integration tests can be

  * general for all instrument, for instance `tests/integration/new/drtsans/test_stitch.py`.
  * specific to a suite of similar instruments, for instance `tests/integration/new/drtsans/mono/test_transmission.py` for all mono-wavelength instruments including Bio-SANS and GP-SANS.
  * specific to an individual instrument, for instance, `tests/integration/new/drtsans/mono/gpsans/test_find_beam_center.py` for GP-SANS and 
    `tests/integration/new/drtsans/tof/eqsans/test_apply_solid_angle.py` for EQ-SANS.

* Testing data location

  Refer to section *Testing data location* in :doc:`README_developer <README_developer>`.

* CI/CD


Glossaries
##########

* SANS
  Small-angle neutron scattering (SANS) is an experimental technique that uses elastic neutron scattering at small scattering angles to investigate the structure of various substances at a mesoscopic scale of about 1–100 nm.
   
   * https://en.wikipedia.org/wiki/Small-angle_neutron_scattering
   * https://www.nist.gov/ncnr/neutron-instruments/small-angle-neutron-scattering-sans

* drtSANS
   
   Data reduction tool for small angle neutron scattering.


Required libraries
##################

* numpy: https://numpy.org/

* Mantid: https://www.mantidproject.org/, https://github.com/mantidproject/mantid

* Others: h5py, docutils, jsonschema, lmfit, matplotlib, mpld3, numexpr, pandas, sortedcontainers, tinydb, ipywidgets

* For unit and integration tests: pytest, pytest-xdist

* For documentation: sphinx, sphinxcontrib-napoleon, 

* For linting and formatting: autopep8, flake8, pylint

