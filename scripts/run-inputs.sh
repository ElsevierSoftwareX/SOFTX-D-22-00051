#! /usr/bin/env bash

for SCRIPT in ${@}; do
  python "${SCRIPT}"
done