# dsm150-2022-oct

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
apt install wget
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh
bash Miniconda3-py38_23.1.0-1-Linux-x86_64.sh
```

Now close and reopen the terminal. You can use `conda list` to test the installation.

### Create jo_wilder env

```bash
 conda create -n jo_wilder python=3.7
 conda activate jo_wilder

 # install dependencies
 conda install -p /root/miniconda3/envs/jo_wilder ipykernel --update-deps --force-reinstall

 pip install pandas tqdm
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
