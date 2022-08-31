FROM    ubuntu:bionic

RUN     apt-get update
RUN     apt-get install -y software-properties-common
RUN     apt-add-repository universe
RUN     apt-get update
RUN     apt-get install -y python3.8
RUN     update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN     apt-get install -y python-pip
RUN     DEBIAN_FRONTEND="noninteractive" apt-get install -y build-essential git libftdi-dev libftdi1 doxygen python3-pip libsdl2-dev curl cmake libusb-1.0-0-dev scons gtkwave libsndfile1-dev rsync autoconf automake texinfo libtool pkg-config libsdl2-ttf-dev

RUN     git clone https://github.com/GreenWaves-Technologies/gap8_openocd.git
WORKDIR /gap8_openocd
RUN     ./bootstrap
RUN     ./configure --program-prefix=gap8- --prefix=/usr --datarootdir=/usr/share/gap8-openocd
RUN     make -j
RUN     make -j install
#RUN     stat /etc/udev/
#RUN     cp /usr/share/gap8-openocd/openocd/contrib/60-openocd.rules /etc/udev/rules.d
#RUN     udevadm control --reload-rules && sudo udevadm trigger
#RUN     usermod -a -G dialout ubuntu

WORKDIR /
RUN     git clone https://github.com/GreenWaves-Technologies/gap_riscv_toolchain_ubuntu_18.git
WORKDIR /gap_riscv_toolchain_ubuntu_18
RUN     ./install.sh /usr/lib/gap_riscv_toolchain

RUN     git clone https://github.com/GreenWaves-Technologies/gap_sdk/
WORKDIR /gap_riscv_toolchain_ubuntu_18/gap_sdk
RUN     git checkout a3dedd5cd8a680a88d2dca2ab7a4ae65cebf4c8d
RUN     pip install -r requirements.txt
RUN     python3 -m pip install -r requirements.txt
SHELL   ["/bin/bash", "-c"] 
RUN     source sourceme.sh && \
        make minimal && \
        make gvsoc
        
WORKDIR /gap_riscv_toolchain_ubuntu_18/gap_sdk/
RUN     git clone https://github.com/pulp-platform/dory
WORKDIR /gap_riscv_toolchain_ubuntu_18/gap_sdk/dory/
RUN     git submodule update --remote --init dory/Hardware_targets/GAP8/Backend_Kernels/pulp-nn-mixed
RUN     git submodule update --remote --init dory/dory_examples
RUN     git submodule update --remote --init dory/Hardware_targets/GAP8/Backend_Kernels/pulp-nn

RUN     python3 -m pip install Cython
RUN     python3 -m pip install --upgrade pip setuptools wheel
RUN     python3 -m pip install setuptools_rust
RUN     python3 -m pip install python-dev-tools --user --upgrade
RUN     python3 -m pip install numpy
RUN     python3 -m pip install onnx
RUN     python3 -m pip install future-fstrings
RUN     python3 -m pip install pandas
RUN     python3 -m pip install ortools
RUN     python3 -m pip install mako
RUN     python3 -m pip install pytest

WORKDIR /gap_riscv_toolchain_ubuntu_18/gap_sdk/dory/
RUN     python3 -m pip install -e .

WORKDIR /gap_riscv_toolchain_ubuntu_18/gap_sdk/

RUN     apt-get install wget

RUN     python3 -m pip uninstall -y torch torchvision
RUN     python3 -m pip install torch==1.6.0 torchvision==0.7.0