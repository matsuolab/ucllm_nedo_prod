#!/bin/bash


nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)


ssh_config_file="${HOME}/.ssh/config"
echo "" > "${ssh_config_file}" 

for node in $nodes; do
    echo "Host $node" >> "${ssh_config_file}"
    echo "    HostName $node" >> "${ssh_config_file}"
    echo "    Port 22" >> "${ssh_config_file}"
    echo "    StrictHostKeyChecking no" >> "${ssh_config_file}"
    echo "" >> "${ssh_config_file}"
done

echo "SSH configuration has been updated."
cat ${ssh_config_file}