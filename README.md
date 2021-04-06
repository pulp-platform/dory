DORY: Deployment ORiented to memorY
===================================

DORY is an automatic tool to deploy DNNs on low-cost MCUs with typically less than 1MB of on-chip SRAM memory. 

### Reference
If you use the DORY tool to deploy your models, please make sure to cite our paper: https://ieeexplore.ieee.org/document/9381618 (preprint available also at https://arxiv.org/abs/2008.07127)
```
@article{burrello2020dory,
  author={A. {Burrello} and A. {Garofalo} and N. {Bruschi} and G. {Tagliavini} and D. {Rossi} and F. {Conti}},
  journal={IEEE Transactions on Computers}, 
  title={DORY: Automatic End-to-End Deployment of Real-World DNNs on Low-Cost IoT MCUs}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TC.2021.3066883}
}
```

Highlights
--------
DORY abstracts tiling as a Constraint Programming~(CP) problem: it maximizes L1 memory utilization under the topological constraints imposed by each DNN layer.
Then, it generates ANSI C code to orchestrate off- and on-chip transfers and computation phases.
Layer tiling is depicted in Fig.1.
<p align="center">
  <img src="images/L3_L2_L1_layer_NEW.png" align="middle" width="1024">
  <br>
  <em> Fig.1 DORY L3-L2-L1 layer routine example. On the left, the I/O DMA copies weights tile in case only Cy is L3-tiled. Two different buffers are used for L2w. Then, the Cluster DMA manages L2-L1 communication using double-buffering, while the cores compute a kernel on the current tile stored in one of the L1 buffers. </em>
</p>


Platform Supported
------------------
The current platform supported is GAP8 v3. Future work will include the support of the latest version of OPEN-PULP and ARM processors.

Limitations
-----------
The DORY framework is currently tested on feed-forward networks with single-wire residual connections. NEMO produces the input ONNXs.
You have to set the "v2" chip flag in DORY parameters to use GAP8 v2 boards or v1 boards. Further, you have to flash weights by using the old pulpbridge manually.

Supported layer types
---------------------
* Pointwise Convolution (+ BatchNorm + Relu)
* DepthWise Convolution (+ BatchNorm + Relu)
* Max Pooling (+ BatchNorm)
* Average Pooling (+ BatchNorm)
* Add (+ BatchNorm + Relu) -- NOT FULLY TESTED
* Linear Layer (+ BatchNorm + Relu)
All layers are implemented in 8-bit integers.
* Linear Layer 32 bits output -- final layer

Topology tested
---------------
* MobilenetV1-128
* Custom networks
* Coming soon: MobilenetV1-224, MobilenetV2-128, MobilenetV2-224

Requirements
------------

### Backend
The DORY framework can be tested using the gvsoc of GAP8 from GreenWaves.
A detailed guide on installing and setting up the latest version can be found at [link](https://greenwaves-technologies.com/manuals/BUILD/HOME/html/index.html#section7).
The DORY tool is tested using 3.6 Realase of gap_sdk, commit: *c6494b97314470446674bb468d31e4391fb187e9* .

### Python
The framework has been developed using python 3.6.8.
The following packages are needed:
* Mako (1.0.12)
* numpy (1.18.4) 
* onnx (1.5.0)    
* pandas (1.0.3)
* ortools (7.5.7466)

### Input
The framework receives as input:
1. an ONNX quantized network generated with the Nemo tool. Refer to [nemo](https://github.com/pulp-platform/nemo) for Nemo framework installation and execution.

Note that only a standard format 8-bit quantized produced by NEMO can be read; key features:
1. Convolution, Pooling or Matmul/Gemm layers supported;
2. Sequences of Mul-Add and Mul-Div recognized as Batch normalization and re-quantization;
Note that other kinds of sequences are not supported (e.g. individual Mul operators).
Examples are given inside [DORY examples](https://github.com/pulp-platform/dory_examples)

Installation
------------
The execution of dory requires the following folders:
1. dory: repository with the framework
2. pulp-nn: repository with backend kernels developed for DORY flow execution

Execute the following commands to clone DORY and pulp-nn backend: 
```
git clone https://github.com/pulp-platform/dory
git submodule update --init --recursive
```

Execution
---------
There are two functions to call to generate a network:
1. extrapolating parameters from ONNX file: 

	*ONNX_management(args).parameters_from_onnx(args)*
2. starting from a custom graph, create the layers and network file to run on PULP

	*model_deploy(args).print_model_network(args)*
	
By correctly running these 2 functions, an application folder is created with all the necessary files.

Examples
--------
To download the examples built on DORY, clone the internal dory_example submodule:
```
cd dory
git submodule update --init --recursive
```
The power profiling on a GAP8 v3 of a 1.0-MobilenetV1-128 is reported in Fig.2.
<p align="center">
  <img src="images/network_power.PNG" align="middle" width="1024">
  <br>
  <em> Fig.2 In the left part, the 1.0-MobileNet-128 power profile when running on GAP-8 @ fc cluster = 100 MHz and VDD = 1V. On the right, number of MAC operations, average power, and time for each layer of the network. Power was sampled at 64 KHz and then filtered with a moving average of 300 micro seconds. </em>
</p>


### Contributors
+ **Alessio Burrello**, *University of Bologna*, [email](mailto:alessio.burrello@unibo.it)
+ **Francesco Conti**, *University of Bologna*, [email](mailto:f.conti@unibo.it)
+ **Angelo Garofalo**, *University of Bologna*, [email](mailto:angelo.garofalo@unibo.it)
+ **Nazareno Bruschi**, *University of Bologna*, [email](mailto:nazareno.bruschi@unibo.it)
+ **Giuseppe Tagliavini**, *University of Bologna*, [email](mailto:giuseppe.tagliavini@unibo.it)
+ **Davide Rossi**, *University of Bologna*, [email](mailto:davide.rossi@unibo.it)
+ **Luca Benini**, *University of Bologna* and *ETH Zurich*, [email](mailto:luca.benini@unibo.it)

### License
DORY is released under Apache 2.0, see the LICENSE file in the root of this repository for details.
