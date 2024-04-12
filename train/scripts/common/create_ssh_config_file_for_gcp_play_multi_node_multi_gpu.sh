#!/bin/bash

set -e
echo ""

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)

ssh_config_file="${HOME}/.ssh/config"
echo "" > "${ssh_config_file}" 

for node in $nodes; do
    # Update the known_hosts file for each node, removing old keys
    ssh-keygen -f "${HOME}/.ssh/known_hosts" -R "$node"
    # Add new node configuration to the SSH configuration file
    echo "Host $node" >> "${ssh_config_file}"
    echo "    HostName $node" >> "${ssh_config_file}"
    echo "    Port 22" >> "${ssh_config_file}"
    echo "    StrictHostKeyChecking no" >> "${ssh_config_file}"
    echo "" >> "${ssh_config_file}"
done

echo "ssh_config_file = ${ssh_config_file}"
echo ""
echo "SSH configuration has been updated."
cat ${ssh_config_file}
echo ""
