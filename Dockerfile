FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y libsm-dev libgl1-mesa-glx python3-pip libyaml-dev

# reset base env
RUN conda install --revision 0 && \
    conda update -n base -c defaults conda

RUN cd /workspace  && git clone -b single-model-load-on-multiple-videos https://github.com/bld-ai/alphapose.git

WORKDIR /workspace/alphapose

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin/:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN conda install python=3.6 pip pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 matplotlib"<3.4" -c pytorch -y && \
    pip install cython

RUN chmod +x "setup.sh" && "./setup.sh"

# include weights, models, and sample data
RUN mkdir detector/yolo/data/ examples/vid/ examples/res/
COPY detector/yolo/data/* detector/yolo/data/
COPY pretrained_models/*.pth pretrained_models/
COPY examples/vid/*.mp4 examples/vid/

ENTRYPOINT ["/bin/bash"]
