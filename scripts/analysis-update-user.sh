#! /bin/bash
set -expu

NDAV_USER='xv6'
NDAV_USER_ID='29091'
NDAV_GROUP='scse_priv'
NDAV_GROUP_ID='55781'
PIP_DIR='/SNS/software/miniconda2/envs/sans'

groupadd -g "${NDAV_GROUP_ID}" "${NDAV_GROUP}"
useradd -s /bin/bash -u "${NDAV_USER_ID}" -m -g "${NDAV_GROUP}" "${NDAV_USER}"

CURRENT_OWNER="$(ls -ld ${PIP_DIR} | awk '{print $3}')"
if [[ ! "${CURRENT_OWNER}" == "${NDAV_USER}" ]]; then
  CURRENT_OWNER_ID="$(ls -lnd ${PIP_DIR} | awk '{print $3}')"
  useradd -s /bin/bash -u "${CURRENT_OWNER_ID}" -m -g "${NDAV_GROUP}" "${CURRENT_OWNER}"
  runuser -l "${CURRENT_OWNER}" -c "chown -R ${NDAV_USER} ${PIP_DIR}"
fi

runuser -l "${NDAV_USER}" -c "bash /opt/sans-backend/scripts/analysis-update.sh $1"
