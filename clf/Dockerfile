FROM nvidia/cuda:10.2-base-ubuntu18.04

# Package version control

ARG PYTHON_VERSION=3.8
ARG CUDA_VERSION=10.2
ARG PYTORCH_VERSION=1.10
ARG CUDA_CHANNEL=nvidia
ARG INSTALL_CHANNEL=pytorch

# Setup workdir and non-root user

ARG USERNAME=hcmus
WORKDIR /home/$USERNAME/workspace/

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
COPY requirements.txt $WORKDIR
RUN conda install -c pytorch -y faiss-gpu && \
    conda init bash && conda activate && \
    python -m pip install -r requirements.txt

USER $USERNAME
RUN conda init bash 
