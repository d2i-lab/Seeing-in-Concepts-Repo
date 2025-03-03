# Installing Faiss

## Standard installs

We support compiling Faiss with cmake from source and installing via conda on a limited set of platforms: Linux (x86 and ARM), Mac (x86 and ARM), Windows (only x86). 
For this, see [INSTALL.md](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md). 

### Why don't you support installing via XXX ? 

The reason why we don't support more platforms is because it is a lot of work to make sure Faiss runs 
in the supported configurations: building the conda packages for a new release of Faiss always surfaces 
compatibility issues. Anaconda provides a sufficiently controlled environment that we can be confident that it will run on 
the user's machines (this is not the case with pip). 
Besides, the platform (hardware and OS) has to be supported by our CI tool (circleCI).

So we are very carful before we add new officially supported platforms (hardware and software). 
We *are* very interested in success (or failure!) stories about porting to other platforms, and related PRs.

## Special configurations

### Compiling the python interface within an Anaconda install

The idea is to install everything via anaconda and link Faiss against that. 
This is useful to make sure the MKL impementation is as fast as possible. 

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate host_env_for_faiss     # an environment that contains python and numpy 

git clone https://github.com/facebookresearch/faiss.git faiss_xx

cd faiss_xx

LD_LIBRARY_PATH= MKLROOT=/private/home/matthijs/anaconda3/envs/host_env_for_faiss/lib CXX=$(which g++) \
$cmake -B build -DBUILD_TESTING=ON -DFAISS_ENABLE_GPU=OFF \
             -DFAISS_OPT_LEVEL=axv2 \
             -DFAISS_ENABLE_C_API=ON \
             -DCMAKE_BUILD_TYPE=Release \
             -DBLA_VENDOR=Intel10_64_dyn .


make -C build -j10 swigfaiss && (cd build/faiss/python ; python3 setup.py build)

(cd tests ; PYTHONPATH=../build/faiss/python/build/lib/ OMP_NUM_THREADS=1 python -m unittest -v discover )

```


### Compiling Faiss on ARM

Commands for an ubuntu 18 image on an Amazon c6g.8xlarge machine : 

```bash

set -e

sudo apt-get install libatlas-base-dev libatlas3-base
sudo apt-get install clang-8
sudo apt-get install swig

# cmake provided with ubuntu is too old

wget https://github.com/Kitware/CMake/releases/download/v3.19.3/cmake-3.19.3.tar.gz

tar xvzf cmake-3.19.3.tar.gz
cd cmake-3.19.3/
./configure --prefix=/home/matthijs/cmake &&  make -j

cd $HOME

alias cmake=$HOME/cmake/bin/cmake

# clone Faiss

git clone https://github.com/facebookresearch/faiss.git

cd faiss

cmake  -B build -DCMAKE_CXX_COMPILER=clang++-8 -DFAISS_ENABLE_GPU=OFF  -DPython_EXECUTABLE=$(which python3) -DFAISS_OPT_LEVEL=generic -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST\
ING=ON

(cd build/faiss/python/ ; python3 setup.py build)

# run tests

export PYTHONPATH=$PWD/build/faiss/python/build/lib/

python3 -m unittest discover


```