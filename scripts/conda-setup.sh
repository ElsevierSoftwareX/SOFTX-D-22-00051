#! /usr/bin/env bash
set -xpe

# the base image should have taken care of these
conda update -n base -c defaults conda
conda config --add channels conda-forge
conda config --add channels mantid
conda install -q -y mamba -n base -c conda-forge
source activate mantid
conda install -q -y -c mantid/label/nightly python=3.7 mantid-framework

# these should be the new additions
conda install -q -y --file /opt/sans-backend/requirements.txt
conda install -q -y --file /opt/sans-backend/requirements_dev.txt

# cleanup
conda clean -afy
python -c "import mantid; print(mantid.__version__)"
