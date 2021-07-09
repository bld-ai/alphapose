# Install AlphaPose

#### Remove build folder
cd SageMaker/AlphaPose/
rm -rf build/
rm -rf alphapose.egg-info

#### Conda environment
conda create -n alphapose gcc_linux-64=5.4.0 gxx_linux-64=5.4.0 python=3.6 pip -y
source activate alphapose
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 matplotlib"<3.4" -c pytorch -y

#### "Upgrade" gcc/g++
mkdir -p $HOME/.local/bin
ln -s $GCC $HOME/.local/bin/gcc
ln -s $GXX $HOME/.local/bin/g++

#### Environment variables
export PATH=$HOME/.local/bin/:/usr/local/cuda/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH

#### Install dependencies
python -m pip install cython gpustat
sudo yum install libyaml-devel

#### Build
python setup.py build develop

#### Test demo_inference
# python scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir examples/demo/