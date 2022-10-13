FROM    ubuntu:bionic

RUN     apt-get update && \
        apt-get install -y software-properties-common && \
        apt-add-repository universe && \
        apt-get update && \
        apt-get install -y python3.8 && \
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
        apt-get install -y python-pip && \
        apt-get install -y python3.8-venv && \
        DEBIAN_FRONTEND="noninteractive" apt-get install -y build-essential git libftdi-dev libftdi1 doxygen python3-pip libsdl2-dev curl cmake libusb-1.0-0-dev scons gtkwave libsndfile1-dev rsync autoconf automake texinfo libtool pkg-config libsdl2-ttf-dev wget unzip graphicsmagick-libmagick-dev-compat sed

#RUN     stat /etc/udev/
#RUN     cp /usr/share/gap8-openocd/openocd/contrib/60-openocd.rules /etc/udev/rules.d
#RUN     udevadm control --reload-rules && sudo udevadm trigger
#RUN     usermod -a -G dialout ubuntu
# GAP-SDK & TOOLCHAIN INSTALLATION
RUN     python3 -m venv /dory_env && \
        source /dory_env/bin/activate && \
        git clone https://github.com/GreenWaves-Technologies/gap8_openocd.git && \
        cd gap8_openocd && \
       ./bootstrap && \
        ./configure --program-prefix=gap8- --prefix=/usr --datarootdir=/usr/share/gap8-openocd && \
        make -j && \
        make -j install && \
        cd / && \
        git clone https://github.com/GreenWaves-Technologies/gap_riscv_toolchain_ubuntu_18.git && \
        cd /gap_riscv_toolchain_ubuntu_18 && \
        ./install.sh /usr/lib/gap_riscv_toolchain && \
        git clone https://github.com/GreenWaves-Technologies/gap_sdk/ && \
        cd gap_sdk && \        
        git checkout a3dedd5cd8a680a88d2dca2ab7a4ae65cebf4c8d && \
        python3 -m pip install -r requirements.txt
SHELL   ["/bin/bash", "-c"]
RUN     source /dory_env/bin/activate && \
        cd /gap_riscv_toolchain_ubuntu_18/gap_sdk && \
        source sourceme.sh && \
        make minimal && \
        make gvsoc && \
# PULP-SDK INSTALLATION
        cd / && \
        git clone https://github.com/pulp-platform/pulp-sdk.git && \
        cd pulp-sdk && \
        source configs/pulp-open-nn.sh && \
        make all && \
# PULP-NN TOOLCHAIN DOWNLOAD
        cd / && \
        wget https://iis-nextcloud.ee.ethz.ch/s/aYESyR5W9FrHgYa/download/riscv-nn-toolchain.zip && \
        unzip riscv-nn-toolchain
# DORY REPO INIT - CI USES THE ${GITHUB_WORKSPACE} VOLUME AT /dory_checkout!!!!
WORKDIR /gap_riscv_toolchain_ubuntu_18/gap_sdk/
SHELL   ["/bin/bash", "-c"]
RUN     source /dory_env/bin/activate && \
        git clone https://github.com/pulp-platform/dory && \
        cd /gap_riscv_toolchain_ubuntu_18/gap_sdk/dory/ && \
        git submodule update --remote --init dory/Hardware_targets/GAP8/Backend_Kernels/pulp-nn-mixed && \
        git submodule update --remote --init dory/dory_examples && \
        git submodule update --remote --init dory/Hardware_targets/GAP8/Backend_Kernels/pulp-nn && \
        python3 -m pip install Cython && \
        python3 -m pip install --upgrade pip setuptools wheel && \
        python3 -m pip install setuptools_rust && \
        python3 -m pip install python-dev-tools --user --upgrade && \
        python3 -m pip install numpy && \
        python3 -m pip install onnx && \
        python3 -m pip install future-fstrings && \
        python3 -m pip install pandas && \
        python3 -m pip install ortools && \
        python3 -m pip install mako && \
        python3 -m pip install pytest && \
        python3 -m pip install Pygments && \
        python3 -m pip install MarkupSafe && \
        cd /gap_riscv_toolchain_ubuntu_18/gap_sdk/dory/ && \
        python3 -m pip install -e . && \
        cd /gap_riscv_toolchain_ubuntu_18/gap_sdk/ && \
        python3 -m pip uninstall -y torch torchvision && \
        python3 -m pip install torch==1.6.0 torchvision==0.7.0
