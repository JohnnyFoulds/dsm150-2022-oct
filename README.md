# dsm150-2022-oct


## From Clean Tensorflow Image

### Install Conda

Perform the following commands in the terminal:

```bash
cd ~
apt update
apt install wget
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh
bash Miniconda3-py38_23.1.0-1-Linux-x86_64.sh
```

Now close and reopen the terminal. You can use `conda list` to test the installation.s

In the VS Code `settings` for the remote tab you need to set the search path to the environment. The result in the settings.json will look as follows. If you can't find `venv` when searching in the settings make sure the Python extension is installed and restart VS Code.

```json
{
    "python.venvPath": "/root/miniconda3/envs/"
}
```

### Create a Pycaret Environment

```bash
# create a conda environment
 conda create -n pycaret python=3.8

 # activate conda environment
 conda activate pycaret

# install pycaret
 pip install pycaret==3.0.0

 # if you want jupyter lab
 conda install -c conda-forge jupyterlab
 python -m ipykernel install --user --name pycaret --display-name "pycaret"
jupyter lab --allow-root

# install tensorflow
pip install tensorflow==2.11.0
pip install tensorflow-addons==0.19.0
pip install keras-tuner==1.3.0

pip install mlflow==2.2.2
pip install seaborn==0.12.2

pip install pandas-profiling==3.6.6
pip install pygwalker==0.1.6.1
pip install dtale==2.13.0
```

Please note that the newest Jupyter extension at the time of writing this is not rendering the HTML output Pycaret. A version confirmed where this is working is `v2022.11.1003412109` which you can downgrade to in the extensions tab.

If they `tqdm` prograss does not work try `pip install ipywidgets==7.6.6`.


# Original Readme

## CUDA Libraries

```bash
sudo apt-get update
sudo apt-get install libcudnn8
#sudo apt-get install libcudnn8-dev

#sudo apt-get install python3-libnvinfer
#sudo apt install libnvinfer8

#pip install nvidia-tensorrt
#sudo apt-get install tensorrt
```

## Graphviz

```bash
sudo apt install graphviz
```

# Cuda 12 Dependencies

```bash
cd ~
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run
```

```bash
sudo apt install libcublas-12-0 #libcublas-dev-12-0
```

## Install Conda

Perform the following commands in the terminal:

```bash
cd ~
apt update
apt install wget
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh
bash Miniconda3-py38_23.1.0-1-Linux-x86_64.sh
```

Now close and reopen the terminal. You can use `conda list` to test the installation.

### Create pycaret env

```bash
 conda create -n pycaret python=3.8
 conda activate pycaret
 pip install pycaret==3.0.0

 # install dependencies
 conda install -p /root/miniconda3/envs/pycaret ipykernel --update-deps --force-reinstall

 pip install pandas tqdm ipywidgets cython
 pip install tensorflow==2.11.0
```

```bash

### Create jo_wilder env

```bash
 conda create -n jo_wilder python=3.7
 conda activate jo_wilder

 # install dependencies
 conda install -p /root/miniconda3/envs/jo_wilder ipykernel --update-deps --force-reinstall

 pip install pandas tqdm ipywidgets cython
 #pip install tensorflow==2.11.0
conda install -c anaconda tensorflow
```

If you used the default settings the path to the environment will be `/root/miniconda3/envs/jo_wilder`.

In the `settings` for the remote tab you need to set the search path to the environment.
The result in the settings.json will look as follows:

```json
{
    "python.venvPath": "/root/miniconda3/envs/"
}
``
