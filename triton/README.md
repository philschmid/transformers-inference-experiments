# Nvidia Triton

Triton Inference Server is a model serving software that simplifies the deployment of AI models at scale in production. It's an open-source serving software that lets teams deploy trained AI models from any framework (TensorFlow, TensorRT, PyTorch, ONNX Runtime, or a custom framework) on any GPU- or CPU-based infrastructure (cloud, data center, or edge). Learn about high performance inference serving with Triton's concurrent execution, dynamic batching, and integrations with Kubernetes and other tools.

**Benefits:**
* Dynamic batches, triton wait 5-10ms to create a dynamic batch and run inference at once 
* Concurrent Model Execution: 
    * Triton can run multiple copies of the same/different model concurrently on the same GPU
* Model Analyzer will optimize the inference performance by specifing contraints on performance results, range of configurations to sweep through.
* KFserving Dataplanev2 Protocol: [Examples](https://github.com/kubeflow/kfserving/tree/master/docs/samples/v1beta1/triton)
* GPU or CPU:
  * GPU Inference TensorRT is recommended
  * CPU Inference ONNX, Pytorch is recommended


**Upcoming:**
* Tooling to automatically convert trained models from different frameworks to TensorRT plan and set up deployment environment

# Resourcecs

* [Docker Docs](https://github.com/triton-inference-server/server/blob/master/docs/compose.md)
* [Python_backend](https://github.com/triton-inference-server/python_backend)
* [Benchmarking](https://blog.einstein.ai/benchmarking-tensorrt-inference-server/)


# Current Example Implementations 

[python `transformers` model](./python_backend)
