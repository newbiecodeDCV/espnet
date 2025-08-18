# Dockerfile
ARG DOCKER_BASE_IMAGE="nvidia/cuda:12.8.0-runtime-ubuntu22.04"
ARG DOCKER_BASE_IMAGE="nvcr.io/nvidia/pytorch:20.12-py3"

FROM $DOCKER_BASE_IMAGE

# password for sudo in docker
ARG PW=1
RUN useradd -m asr --uid=1011 && echo "asr:1" | chpasswd
USER 1011:997
