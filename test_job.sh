#!/bin/bash

set -x

TEST_SCOPE=$1  # 'unit' or 'integration'

# the default environment in bash should already be drtsans-dev
source activate drtsans-dev
cd /opt/sans-backend
echo "Writing tests results to $(pwd)/${TEST_SCOPE}_test_results.xml"
pytest --dist loadscope -v /opt/sans-backend/tests/${TEST_SCOPE} -n 4 --junitxml=./${TEST_SCOPE}_test_results.xml
