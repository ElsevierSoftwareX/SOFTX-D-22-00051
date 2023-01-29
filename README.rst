========
drt-sans
========

Data Reduction Toolkit for Small Angle Neutron Scattering

This packages is a collection of functionality for reducing SANS data developed in collaboration with the instrument scientists at the High Flux Isotope Reactor (HFIR) and Spallation Neutron Source (SNS) at Oak Ridge National Laboratory.

While much of the functionality is generic, this implementation is aimed at reducing data from BIOSANS, EQSANS, and GPSANS.
As appropriate, this work is an abstraction layer on top of the mantid project.

**This is a python3 only package.**

------------------------------
Usage from provided front-ends
------------------------------

For end users go to
`QA version <http://scse-ui.ornl.gov:8080/>`_

Use `jupyter <https://jupyter.sns.gov/>`_ to have a play with the code.
The kernel to select is ``sans at ...``.

One can run scripts directly on `analysis <https://analysis.sns.gov/>`_ cluster.
To do that, open a terminal and activate the desired conda environment. The options are:

* ``sans`` the latest stable release
* ``sans-qa`` the future stable release (to be tested right before the next iteration)
* ``sans-dev`` the latest development version

The easiest way to start an interactive ipython session is by running

.. code-block:: shell

   $ drtsans

adding ``--qa`` or ``--dev`` will start the qa or development version respectively.
The ``drtsans`` wrapper script launches ipython with the selected conda environment located in ``/opt/anaconda/envs/` and deactivates the conda environment when the session ends.


One must have an XCAMS account to use either the jupyter kernel provided above.

-------------------------------------
Using the Docker packaged environment
-------------------------------------

This the instructions for someone who wants to use the Docker container
created through the automated build pipeline to develop drt-sans, use
drt-sans to develop reduction scripts, or test existing drt-sans
functionality. The SNS analysis cluster does not have Docker installed
and Docker is required to follow these instructions.

1. (If not installed) `Install Docker <https://docs.docker.com/install/>`_
2. Download the latest ``sans-backend-run.sh`` `script <scripts/sans-backend-run.sh>`_ from the feature, release, or master branch for which you are testing:
3. Run the script with ``sudo bash sans-backend-run.sh -h`` to see the help menu.

Current options include:

* ``-i`` launches a bash shell
* ``-u`` forces an update of the application
* ``-h`` prints the help message

You must download the wrapper script from the above link as the build process modifies the copy in version control.

-----------------------------------------------
Set-up for development in a virtual environment
