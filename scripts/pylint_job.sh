#!/bin/bash
set -x

source activate drtsans-dev
cd /opt/sans-backend
python -m pylint --exit-zero --disable=C --disable=fixme --disable=no-name-in-module drtsans tests --ignore drtsans/_version.py
