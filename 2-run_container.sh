#!/bin/bash

# Docker GPU Runner Script
#
# This script simplifies running Docker containers with NVIDIA GPU support
# for ASR (Automatic Speech Recognition) development. It automatically handles:
# - GPU allocation
# - Volume mounting
# - Environment variables
# - Proxy settings
# - Container naming with timestamp
# - Cleanup on interrupt
#
# Docker image configuration
image="nvcr.io/nvidia/cuda"
tag="12.8.0-runtime-ubuntu22.04"

# image="nvcr.io/nvidia/pytorch"
# tag="20.12-py3-asr"


echo "Using image ${image}:${tag}."

# Generate timestamp for unique container naming
this_time="$(date '+%Y%m%dT%H%M')"

# Determine the appropriate docker command with GPU support
if [ -z "$(which nvidia-docker)" ]; then
    # Use docker with --gpus option if nvidia-docker not available
    cmd="docker run --gpus all"
else
    # Use nvidia-docker with specific GPU allocation
    cmd="NV_GPU='${docker_gpu}' nvidia-docker run"
fi

# Base docker run command with common options
cmd="${cmd} -it --rm -p 6006:6006 --net=host --ipc=host \
    -w /speech-classification \
    -v /storage/asr/hiennt/Speechdw:/speech-classification \
    -v /storage/asr/data/LibriSpeech:/data"

# Initialize environment variables string
this_env=""

# Process additional environment variables if provided
if [ ! -z "${docker_env}" ]; then
    # Split comma-separated env variables
    docker_env=$(echo ${docker_env} | tr "," "\n")
    for i in ${docker_env[@]}
    do
        this_env="-e $i ${this_env}"
    done
fi

# Add proxy settings if they exist in the host environment
if [ ! -z "${HTTP_PROXY}" ]; then
    this_env="${this_env} -e 'HTTP_PROXY=${HTTP_PROXY}'"
fi

if [ ! -z "${http_proxy}" ]; then
    this_env="${this_env} -e 'http_proxy=${http_proxy}'"
fi

# Generate unique container name with GPU info and timestamp
container_name="hiennt_speech_gpu${docker_gpu//,/_}_${this_time}"

# Finalize the docker command
cmd="${cmd} ${this_env} --name ${container_name} ${image}:${tag}"

# Define interrupt handler for clean container removal
trap ctrl_c INT

function ctrl_c() {
    # Handle CTRL-C by forcefully removing the running container
    echo "** Kill docker container ${container_name}"
    docker rm -f ${container_name}
}

echo "Executing application in Docker"
echo ${cmd}
eval ${cmd}

echo "`basename $0` done."