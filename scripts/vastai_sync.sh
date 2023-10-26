#!/bin/bash

INSTANCE_ID=$1

# Fetch instance data
INSTANCE_DATA=$(vastai show instance $INSTANCE_ID --raw)

# Extract SSH port and host
SSH_PORT=$(echo $INSTANCE_DATA | jq -r '.ssh_port')
SSH_HOST=$(echo $INSTANCE_DATA | jq -r '.ssh_host')

SOURCE_DIR="/data/output/"
DEST_DIR="/data/output/vast_${INSTANCE_ID}"

while true; do
    rsync -arzu -v --progress --rsh=ssh -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no" --exclude '*model_states.pt' --exclude '*optim_states.pt' root@${SSH_HOST}:${SOURCE_DIR} ${DEST_DIR}
    sleep 300
done