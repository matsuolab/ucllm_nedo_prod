#!/bin/bash

set -e
echo ""

# Creates a SSH config file.
ssh_config_file="${HOME}/.ssh/config"
while read -r line
do
  echo "Host ${line}"
  echo "    HostName ${line}"
  echo "    Port 2222"
  echo "    StrictHostKeyChecking no"
  echo ""
done < "${SGE_JOB_HOSTLIST}" > "${ssh_config_file}"
echo "ssh_config_file = ${ssh_config_file}"
cat ${ssh_config_file}
echo ""
