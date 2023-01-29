#! /bin/bash
set -expu

PYONCAT_VERSION=1.3.2

case "$1" in

    master) CONDA_ENV='sans'     # `master` branch only
            ;;
    qa)     CONDA_ENV='sans-qa'  # versioned branch to make release candidates from
            ;;
    next)   CONDA_ENV='sans-dev' # `next` branch
            ;;
esac

set +u
export PATH=/SNS/software/miniconda2/bin:$PATH
source activate ${CONDA_ENV}
conda install -q -y -c mantid/label/nightly mantid-framework
pip install /opt/sans-backend

# PyONCat install
CURRENT_PYONCAT_VERSION_INSTALLED=$(conda list pyoncat | grep pyoncat | tr -s ' ' ' ' | cut -d' ' -f 2)
if [ ! "$CURRENT_PYONCAT_VERSION_INSTALLED" == "$PYONCAT_VERSION" ]
then
    pip install https://oncat.ornl.gov/packages/pyoncat-${PYONCAT_VERSION}-py2.py3-none-any.whl
fi
