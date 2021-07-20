FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y libsm-dev libgl1-mesa-glx python3-pip libyaml-dev && \
    conda update -n base -c defaults conda && \
    git clone -b single-model-load-on-multiple-videos https://github.com/bld-ai/alphapose.git

WORKDIR /workspace/alphapose

RUN mkdir detector/yolo/data/

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin/:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

COPY detector/yolo/data/* detector/yolo/data/
COPY pretrained_models/*.pth pretrained_models/
COPY setup.sh /

RUN conda install torchvision==0.3.0 matplotlib"<3.4" -c pytorch -y && \
    pip install scikit-build cmake cython

RUN chmod +x "/setup.sh" && \
    "/setup.sh" && \
    mkdir -p /opt/ml/processing/input/ && \
    mkdir -p /opt/ml/processing/output/

ENTRYPOINT ["/bin/bash"]