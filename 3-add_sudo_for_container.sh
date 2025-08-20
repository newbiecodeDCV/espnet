#!/bin/bash

# Script: add_sudo_for_user_asr.sh
# Description: This script adds sudo privileges for user 'asr' in a specified Docker container.
#              It first checks if a container ID/name is provided as argument.
#              If no argument is provided, it displays usage instructions and lists running containers.
# Usage: bash add_sudo_for_user_asr.sh [container_id_or_name]

# Check if container ID/name argument is provided
if [ -z "$1" ]; then
    # Display usage instructions if no argument is provided
    echo "==========================================================================================="
    echo "Error: Container ID/name must not be empty!"
    echo "Usage: bash add_sudo_for_user_asr.sh [container_id_or_name]"
    echo "=====================================LIST CONTAINER========================================"

    # List all running Docker containers to help user identify the correct container
    docker ps

    echo "==========================================================================================="
else
    # If container ID/name is provided, proceed with setting up sudo for user 'asr'
    echo "Setting up sudo for user 'asr' in Docker container: $1 ..."

    # Execute commands inside the container as root to:
    # 1. Add 'asr' user to sudoers file
    # 2. Update package lists
    # 3. Install sudo package (if not already installed)
    docker exec -it --user root $1 /bin/bash -c "echo '%asr ALL=(ALL:ALL) ALL'>>/etc/sudoers && apt update && apt install sudo"

    # Display completion message
    echo "Done. User 'asr' now has sudo privileges in container: $1"
    echo "Attaching to container: $1"

    # Attach to the container after setup
    docker attach "$1"
fi