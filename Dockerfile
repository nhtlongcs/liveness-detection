FROM nvidia/cuda:11.6.1-base-ubuntu20.04

# Package version control

ARG PYTHON_VERSION=3.8
ARG CUDA_VERSION=11.6
ARG PYTORCH_VERSION=1.12.0
ARG CUDA_CHANNEL=nvidia
ARG INSTALL_CHANNEL=pytorch

# Setup workdir and non-root user

ARG USERNAME=dcu
WORKDIR /home/$USERNAME/workspace/

ENV TZ=Asia/Ho_Chi_Minh \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update &&\
    apt-get install -y --no-install-recommends curl git sudo &&\
    useradd --create-home --shell /bin/bash $USERNAME &&\
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME &&\
    chmod 0440 /etc/sudoers.d/$USERNAME &&\
    rm -rf /var/lib/apt/lists/*


RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get -qq update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    gcc \
    tmux \
    libjpeg-dev \
    unzip bzip2 ffmpeg libsm6 libxext6 \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/*


# # Install conda
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    chown -R $USERNAME:$USERNAME /opt/conda/ && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -c "${INSTALL_CHANNEL}" -c "${CUDA_CHANNEL}" -y \
    python=${PYTHON_VERSION} \
    pytorch=${PYTORCH_VERSION} torchvision "cudatoolkit=${CUDA_VERSION}" && \
    /opt/conda/bin/conda clean -ya

# Set up environment variables
ENV PATH /opt/conda/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PYTORCH_VERSION ${PYTORCH_VERSION}

# # # Install repo dependencies 
COPY ./* $WORKDIR
RUN conda env create -f environment.yml && \
    conda clean -ya

USER $USERNAME
RUN conda init bash
RUN echo "conda activate zaloai" >> ~/.bashrc 
