DORY: Deployment ORiented to memorY
===================================

This is the official public repository for the DORY tool.

Abstract
--------
DORY is an automatic tool to deploy DNNs on low cost MCUs with typically less than 1MB of on-chip SRAM memory. 
DORY abstracts tiling as a Constraint Programming~(CP) problem: it maximizes L1 memory utilization under the topological constraints imposed by each DNN layer.
Then, it generates ANSI C code to orchestrate off- and on-chip transfers and computation phases.
The current implementation supports PULP RISC-V backend.

Layer supported
---------------
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
A detailed guide on how to install and set up the latest version can be found at [link](https://greenwaves-technologies.com/manuals/BUILD/HOME/html/index.html#section7).
The DORY tool is tested using 3.6 Realase of gap_sdk, commit: *c6494b97314470446674bb468d31e4391fb187e9* .

### Python
The framework has been developed using python 3.6.8.
The following packages are needed:
* Mako (1.0.12)
* numpy (1.18.4) 
* onnx (1.5.0)  
* torch (1.5.0)   
* pandas (1.0.3)
* ortools (7.5.7466)

### Input
The framework receives as input:
1. an onnx quantized network generated with the Nemo tool. Refer to [nemo](https://github.com/pulp-platform/nemo) for Nemo framework installation and execution.

Note that only a standard format 8-bit quantized produced by nemo can be read; key features:
1. Convolution, Pooling or Matmul/Gemm layers supported;
2. Sequences of Mul-Add and Mul-Div recognized as batchnorm and requantization;
Note that other kind of sequences are not supported (e.g. individual Mul operators).
Examples are given inside [DORY examples](https://github.com/pulp-platform/dory_examples)

Installation
------------
The execution of dory requires the following folders:
1. dory: repository with framework
2. backend: backend kernels developed for DORY flow execution

Execute the following commands to clone DORY: 
```
git clone https://github.com/pulp-platform/dory
```

Execution
---------
There are 2 functions to call to generate a network:
1. extrapolating parameters from onnx file: 

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


### Reference
If you use the DORY tool to deploy your models, please make sure to cite ...
\citation to be defined