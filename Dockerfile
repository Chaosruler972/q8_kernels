FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt upgrade -y
RUN apt-get install -y lsb-release wget software-properties-common

RUN lsb_release -c | cut -f2 > /root/ubuntu_codename

RUN apt update
RUN wget https://repo.radeon.com/amdgpu-install/6.2.3/ubuntu/`cat /root/ubuntu_codename`/amdgpu-install_6.2.60203-1_all.deb
RUN apt install -y ./amdgpu-install_6.2.60203-1_all.deb

RUN amdgpu-install -y --usecase=workstation,rocm --accept-eula

ARG PYTHON_VERSION="3.10"
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt install python${PYTHON_VERSION} python${PYTHON_VERSION}-venv -y
RUN python${PYTHON_VERSION} -m venv /root/venv
RUN /root/venv/bin/python -m pip install -U pip

RUN wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.3/torch-2.3.0%2Brocm6.2.3-cp310-cp310-linux_x86_64.whl
RUN wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.3/torchvision-0.18.0%2Brocm6.2.3-cp310-cp310-linux_x86_64.whl
RUN wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.3/pytorch_triton_rocm-2.3.0%2Brocm6.2.3.5a02332983-cp310-cp310-linux_x86_64.whl
RUN /root/venv/bin/python -m pip uninstall torch torchvision pytorch-triton-rocm
RUN /root/venv/bin/python -m pip install torch-2.3.0+rocm6.2.3-cp310-cp310-linux_x86_64.whl torchvision-0.18.0+rocm6.2.3-cp310-cp310-linux_x86_64.whl pytorch_triton_rocm-2.3.0+rocm6.2.3.5a02332983-cp310-cp310-linux_x86_64.whl
RUN rm -rf torch-2.3.0+rocm6.2.3-cp310-cp310-linux_x86_64.whl
RUN rm -rf torchvision-0.18.0+rocm6.2.3-cp310-cp310-linux_x86_64.whl
RUN rm -rf pytorch_triton_rocm-2.3.0+rocm6.2.3.5a02332983-cp310-cp310-linux_x86_64.whl

RUN /root/venv/bin/python -m pip install -U build wheel setuptools

WORKDIR /workspace
COPY ./_inner_build_script_docker.sh /root/_inner_build_script_docker.sh
RUN chmod +x /root/_inner_build_script_docker.sh

# Installing CUDA for compatbility, remove thrust
RUN apt install nvidia-cuda-toolkit -y
# Make thrust unnreachable
RUN mv /usr/include/thrust/ /usr/include/thrust.bak
CMD [ "/root/_inner_build_script_docker.sh"]