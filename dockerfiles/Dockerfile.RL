FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

RUN sed -i -e "s/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g" /etc/apt/sources.list && \
    sed -i -e "s/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g" /etc/apt/sources.list && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && apt-get update -y && \
    apt-get install --assume-yes --fix-missing build-essential && \
    apt-get install -y openssh-server vim git curl tmux && \
    apt-get install -y libgl1-mesa-glx libegl-dev && \
    apt-get install -y libegl1-mesa libegl1-mesa-dev libegl1 && \
    rm -rf /var/lib/apt/lists/* && apt-get clean

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_25.5.1-1-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

RUN /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    /opt/conda/bin/conda config --set show_channel_urls yes && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple


WORKDIR /app
COPY LIBERO /app/libero

RUN mkdir -p /app/dexbotic && touch /app/dexbotic/__init__.py
COPY pyproject.toml /app/pyproject.toml

RUN /opt/conda/bin/conda create -n dexbotic-rl python=3.10 -y

# Install LIBERO dependencies
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '%s\n' \
    '{' \
    '    "file_format_version" : "1.0.0",' \
    '    "ICD" : {' \
    '        "library_path" : "/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0"' \
    '    }' \
    '}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN /bin/bash -c "source activate dexbotic-rl && \
    cd libero && \
    pip install -e . && \
    pip install robosuite==1.4.0 numpy==1.26.4 opencv-python-headless==4.11.0.86 && \
    pip install termcolor bddl==1.0.1 future==0.18.2 cloudpickle==2.1.0 gym==0.25.2 imageio && \
    cd .."

# Install dexbotic dependencies
RUN /bin/bash -c "source activate dexbotic-rl && \
        pip install torch==2.2.2 torchvision==0.17.2 xformers --index-url https://download.pytorch.org/whl/cu118 && \
        pip install -e . && \
        pip install  deepspeed==0.17.1 transformers==4.51.0"

RUN /bin/bash -c "source activate dexbotic-rl && \
        pip install ninja && \
        pip install packaging && \
        MAX_JOBS=32 pip install flash-attn --no-build-isolation"

RUN /opt/conda/bin/conda init bash

CMD ["bash"]