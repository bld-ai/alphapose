# make sure we have gcc
sudo apt update
sudo apt install build-essential

# the repository assumes an ubuntu 20.04 system
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list'

# https://developer.nvidia.com/blog/cuda-11-1-introduces-support-rtx-30-series/
sudo apt update
sudo apt install cuda-toolkit-11-1

# build and run cuda samples
cd /usr/local/cuda/samples/4_Finance/BlackScholes
sudo make
./BlackScholes
cd ~

# install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh

# >>> Alphapose Installation >>>
git clone https://github.com/MVIG-SJTU/AlphaPose.git
cd AlphaPose

# we need this pull request to work with pytorch 1.5+
git pull origin pull/592/head

# match conda cudatoolkit with system install
# currently, cudatoolkit 11.1 is the latest version with stable runs on pytorch (11.2, 11.3, 11.4 untested)
# https://pytorch.org/get-started/previous-versions/#v180
conda create -n alphapose python=3.7 -y
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge

# verify that torch can access cuda
python -c "import torch; print(torch.cuda.is_available())"

# add pip requirements and build
export PATH=/usr/local/cuda/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
python -m pip install cython
sudo apt install libyaml-dev
python setup.py build develop
# <<< Alphapose Installed <<<