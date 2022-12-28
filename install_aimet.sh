apt-get update
apt-get install python3.8 python3.8-dev python3-pip
python3 -m pip install --upgrade pip
apt-get install --assume-yes wget gnupg2

# install GPU
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
# mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
# wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda-repo-ubuntu1804-11-1-local_11.1.1-455.32.00-1_amd64.deb
# dpkg -i cuda-repo-ubuntu1804-11-1-local_11.1.1-455.32.00-1_amd64.deb
# apt-key add /var/cuda-repo-ubuntu1804-11-1-local/7fa2af80.pub
# echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list
# echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
# apt-get update
# apt-get -y install cuda

# wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
# dpkg -i nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
# apt-get update


export AIMET_VARIANT=torch_cpu
export release_tag=1.23.0
export download_url="https://github.com/quic/aimet/releases/download/${release_tag}"
export wheel_file_suffix="cp38-cp38-linux_x86_64.whl"

python3.8 -m pip install ${download_url}/AimetCommon-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}

# Install ONE of the following depending on the variant
python3.8 -m pip install ${download_url}/AimetTorch-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix} -f https://download.pytorch.org/whl/torch_stable.html
python3.8 -m pip install ${download_url}/Aimet-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}

cat /usr/local/lib/python3.8/dist-packages/aimet_common/bin/reqs_deb_common.txt | xargs apt-get --assume-yes install

# GPU
# cat /usr/local/lib/python3.8/dist-packages/aimet_torch/bin/reqs_deb_torch_gpu.txt | xargs apt-get --assume-yes install

python3.8 -m pip uninstall -y pillow
python3.8 -m pip install --no-cache-dir Pillow-SIMD==7.0.0.post3

ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /usr/lib

# # If you installed the CUDA 11.x drivers
# ln -s /usr/local/cuda-11.0 /usr/local/cuda
# # OR if you installed the CUDA 10.x drivers
# ln -s /usr/local/cuda-10.0 /usr/local/cuda


# source /usr/local/lib/python3.8/dist-packages/aimet_common/bin/envsetup.sh