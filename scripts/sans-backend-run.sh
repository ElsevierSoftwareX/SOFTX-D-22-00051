#! /bin/bash

set -o errexit -o noclobber -o nounset
VAR_APP_NAME="${0}"
VAR_INTERACT='false'
VAR_UPDATE='false'
VAR_MOUNT_SNS=''

func_help_message() {
  printf "  %s is a wrapper for the ORNL Small Angle Neutron Scattering mantid module.

  Usage: %s [-iuh] param.json
      -i) launches a bash shell.
      -u) forces an update of the application.
      -h) prints this message.\n" "${VAR_APP_NAME}" "${VAR_APP_NAME}"
}

func_chk_perms() {
  if [ ! "$(id -nG | grep -o docker)" ]; then
    if [ "$(id -u)" -ne 0 ]; then
      printf "Error: You must either be in the docker group or run as root (sudo)." 1>&2
      exit 1
    fi
  fi
}

func_main() {
  while getopts "iu-h" OPT; do
    case "${OPT}" in
        i)
            VAR_INTERACT='true';;
        u)
            VAR_UPDATE='true';;
        h)
            func_help_message
            exit 0;;
        -)
            shift
            break;;
        *)
            func_help_message
            printf "Error: Not implemented: %s\n" "${1}" >&2
            exit 1;;
    esac
    shift
  done
  if [[ $# -gt 1 ]]; then
    printf "Error: This script only accepts one parameters file.\n" 1>&2
  elif [[ $# -eq 1 ]]; then
    declare -r VAR_PARAMS_FILE="${1}"
  else
    declare -r VAR_PARAMS_FILE='/tmp/input/scripts/test_input.json'
  fi
  if docker -v 1>/dev/null 2>/dev/null; then
    if [ "${VAR_UPDATE}" = 'true' ]; then
      docker pull ${CONTAINER_URL}
    fi
    if docker login code.ornl.gov:4567 2>/dev/null; then
      if [[ -d /SNS ]] && [[ $(ls -1q /SNS | wc -l) -gt 0 ]]; then
        declare -r VAR_MOUNT_SNS='-v /SNS:/SNS'
      fi
      mkdir -m a=rwx -p SANS_output
      if ${VAR_INTERACT}; then
        docker run -v "$PWD":/tmp/input -v "$PWD"/SANS_output:/tmp/SANS_output ${VAR_MOUNT_SNS} -it ${CONTAINER_URL} bash
      else
        docker run -v "$PWD":/tmp/input -v "$PWD"/SANS_output:/tmp/SANS_output ${VAR_MOUNT_SNS} -t ${CONTAINER_URL} bash -c "source activate drtsans-dev && python3 /tmp/input/scripts/process_reduction.py ${VAR_PARAMS_FILE}"
      fi
    else
      printf "Error: Login failed. Do you have access to this repository?\n" 1>&2
      exit 1
    fi
  else
    printf "Error: Docker doesn't seem to be working. Is it installed?\n" 1>&2
    exit 1
  fi
}

func_chk_perms
func_main ${@}
