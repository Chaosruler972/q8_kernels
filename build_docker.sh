#!/bin/bash

SCRIPT=`realpath $0`
SCRIPT_DIR=`dirname $SCRIPT`

# Build the docker image
docker buildx build --platform linux/amd64 -t q8_kernels_rocm_builder $SCRIPT_DIR

docker run --rm -it -v $SCRIPT_DIR:/workspace q8_kernels_rocm_builder