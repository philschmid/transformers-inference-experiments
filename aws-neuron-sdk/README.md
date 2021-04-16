# AWS Neuron sdk

AWS Neuron is the SDK for [AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/), the custom designed machine learning chips enabling high-performance deep learning inference applications on [EC2 Inf1 instances](https://aws.amazon.com/ec2/instance-types/inf1/). Neuron includes a deep learning compiler, runtime and tools that are natively integrated into TensorFlow, PyTorch and MXnet. With Neuron, you can develop, profile, and deploy high-performance inference applications on top of `[EC2 Inf1 instances](https://aws.amazon.com/ec2/instance-types/inf1/).

**Resources**
* [Whats new/Release notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/index.html#neuron-whatsnew)
* [Github Repository](https://github.com/aws/aws-neuron-sdk)
* [Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-intro/get-started.html)
* [Hugging Face Transformers Example](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/pytorch-neuron/tutorials/tutorial-torchserve.html)
* [Installation Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-intro/install-pytorch.html?highlight=install)
* [transformers optimization documentation](https://huggingface.co/transformers/serialization.html?highlight=onnx)
* [Medium article "PyTorch JIT and TorchScript"](https://towardsdatascience.com/pytorch-jit-and-torchscript-c2a77bac0fff)
* [Medium: Customer Story of Autodesk using Inferentia](https://jineemocha.medium.com/how-we-used-aws-inferentia-to-boost-pytorch-nlp-model-performance-by-4-9x-9f79f5314ca8)
* [AWS Neuron SDK for Performance Tuning](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/perf/performance-tuning.html?highlight=matmul#mixed-precision)
# Getting started:

Currently there is no version of `neuron-sdk` for Mac therefore I created a `dockerfile` to use it in a virtual environment. 

There are two examples one from AWS (`aws_tutorial_bert.ipynb`) and one I created (`simple_mrpc_example.ipynb`) while working with `aws-neuron-sdk`. 


## Setup Dev environment

```Bash
make build
```
## Compile model

Currently, the model is hardcoded into `compile_model.py`

```bash
python compile_model.py
```



# Deploy

TBD.

# Performance Tuning

This section describes the different ways of how you can tune performance for your model using the `aws-neuron-sdk`. You can find all code examples for `transformers` in [performance_tuning.ipynb](./performance_tuning.ipynb)


## Batching and pipelining

**batching** it is achieved by loading the data into an on-chip cache and reusing it multiple times for multiple different model-inputs.  
=> batching is preferred for applications that aim to optimize throughput and cost at the expense of latency.  

**pipelining** this is achieved by caching all model parameters into the on-chip cache across multiple NeuronCores and streaming the calculation across them.
=>  pipelining is preferred for applications with high-throughput requirement under a strict latency budget.

## Mixed Precision

Reduced precision data-types are typically used to improve performance. In the example below, we convert all operations to FP16. Neuron also supports conversion to a mixed-precision graph, wherein only the weights and the data inputs to matrix multiplies and convolutions are converted to FP16, while the rest of the intermediate results are kept at FP32.  
To selectively cast only inputs to MatMul and Conv operators, use option `--fp32-cast=matmult`. This option may be required in certain networks such as BERT where additional accuracy is desired.




# Benchmark tests

![test](./test.png)
