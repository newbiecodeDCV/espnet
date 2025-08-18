#!/bin/bash

# ==============================================
# Docker Image Builder with User Context
#
# This script automates the process of building a Docker image while preserving
# the current user's context (username, UID, GID) as build arguments.
#
# It performs three main steps:
#   1. Displays current user information
#   2. Creates a modified Dockerfile with user context as build arguments
#   3. Builds the Docker image using the modified Dockerfile
#
# Usage:
#   ./script.sh  # Make sure the original Dockerfile exists in the same directory
# ==============================================

# Display current user information
echo "=======GET USER INFO========"
echo "+ username: $(id -nu)"
echo "+ user_id: $(id -nu | id -u)"
echo "+ group_id: $(id -ng | id -g)"
echo "============================"

# Create a new Dockerfile with user context as build arguments
# Strategy:
# 1. Keep first 3 lines of original Dockerfile
# 2. Add ARG lines for USER, UID, GID
# 3. Append remaining lines from original Dockerfile (starting from line 4)
head -n 3 Dockerfile > Dockerfile.new
echo "ARG USER=`id -nu`" >> Dockerfile.new
echo "ARG UID=`id -nu|id -u`" >> Dockerfile.new
echo "ARG GID=`id -ng|id -g`" >> Dockerfile.new
tail -n +4 Dockerfile >> Dockerfile.new

# Build the Docker image using the modified Dockerfile
echo "======BUILD DOCKER IMAGE======"
docker build -f Dockerfile.new . -t "nvcr.io/nvidia/cuda:12.8.0-runtime-ubuntu22.04"
# docker build -f Dockerfile.new . -t "nvcr.io/nvidia/pytorch:20.12-py3-asr"

# Clean up - remove the temporary Dockerfile
rm -v Dockerfile.new