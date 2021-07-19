FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y libsm-dev libgl1-mesa-glx python3-pip libyaml-dev && \
    conda update -n base -c defaults conda && \
    git clone -b single-model-load-on-multiple-videos https://github.com/bld-ai/alphapose.git

WORKDIR /workspace/alphapose

RUN conda create -n alphapose python=3.6 pip -y && \
    mkdir detector/yolo/data/

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin/:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

COPY detector/yolo/data/* detector/yolo/data/
COPY pretrained_models/*.pth pretrained_models/
COPY setup.sh entrypoint.sh /

RUN chmod +x "/setup.sh" && \
    chmod +x "/entrypoint.sh" && \
    echo "conda activate alphapose" >> ~/.bashrc

SHELL ["conda", "run", "--no-capture-output", "-n", "alphapose", "/bin/bash", "-c"]
RUN conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 matplotlib"<3.4" -c pytorch -y && \
    pip install cython && \
    "/setup.sh"

ENTRYPOINT ["/entrypoint.sh"]