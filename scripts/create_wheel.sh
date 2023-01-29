#!/bin/bash
set -x

source activate drtsans-dev
cp -R /opt/sans-backend /tmp/
cd /tmp/sans-backend
python -m build --wheel --no-isolation
check-wheel-contents dist/drtsans-*.whl
