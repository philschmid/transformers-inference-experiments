# AWS Neuron sdk

AWS Neuron is the SDK for [AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/), the custom designed machine learning chips enabling high-performance deep learning inference applications on [EC2 Inf1 instances](https://aws.amazon.com/ec2/instance-types/inf1/). Neuron includes a deep learning compiler, runtime and tools that are natively integrated into TensorFlow, PyTorch and MXnet. With Neuron, you can develop, profile, and deploy high-performance inference applications on top of `[EC2 Inf1 instances](https://aws.amazon.com/ec2/instance-types/inf1/).

**Resources**
* [Whats new/Release notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/index.html#neuron-whatsnew)
* [Github Repository](https://github.com/aws/aws-neuron-sdk)
* [Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-intro/get-started.html)
* [Hugging Face Transformers Example](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/pytorch-neuron/tutorials/tutorial-torchserve.html)

# Getting started:

Currently there is no version of `neuron-sdk` for mac. 

## Install Packages

```bash
# Add Neuron Conda channel to Conda environment
conda config --env --add channels https://conda.repos.neuron.amazonaws.com
```
```bash
# If you are installing Torch-Neuron plus Neuron-Compiler
conda install torch-neuron
```

## Compile model



# Run locally


# Deploy