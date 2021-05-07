
# ONNX

[ONNX](http://onnx.ai/) is open format for machine learning models. It allows to save your neural network's computation graph in a framework agnostic way, which might be particulary helpful when deploying deep learning models.
    
Indeed, businesses might have other requirements _(languages, hardware, ...)_ for which the training framework might not be the best suited in inference scenarios. In that context, having a representation of the actual computation graph that can be shared accross various business units and logics across an organization might be a desirable component.

"Along with the serialization format, ONNX also provides a runtime library which allows efficient and hardware specific execution of the ONNX graph. This is done through the [onnxruntime](https://microsoft.github.io/onnxruntime/) project and already includes collaborations with many hardware vendors to seamlessly deploy models on various platforms.

Through this notebook we'll walk you through the process to convert a PyTorch or TensorFlow transformers model to the [ONNX](http://onnx.ai/) and leverage [onnxruntime](https://microsoft.github.io/onnxruntime/) to run inference tasks on models from  🤗 __transformers__

**Resources**
* [transformers optimization documentation](https://huggingface.co/transformers/serialization.html?highlight=onnx)
* [Medium: Accelerate your NLP pipelines using Hugging Face Transformers and ONNX Runtime](https://medium.com/microsoftazure/accelerate-your-nlp-pipelines-using-hugging-face-transformers-and-onnx-runtime-2443578f4333)
* [ONNX documentation: Transformer Model optimization Tool Overview](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/transformers)
* [Example Notebook for 🤗 Transformers](https://github.com/huggingface/transformers/blob/master/notebooks/04-onnx-export.ipynb)
* [ONNX x Pytorch Opset Version Table](https://github.com/onnx/onnx/blob/master/docs/Versioning.md#released-versions)
* [Sentence Transformers ONNX issue](https://github.com/UKPLab/sentence-transformers/issues/631)
* [ONNX example for Fill-Mask](https://neurocode.io/blog/serverless-nlp-transformer-model-with-onnx-and-azure-functions)

## Setup Dev environment

```Bash
make build
```

```Bash
make start
```

## Compile model

Currently, the model is hardcoded into `compile_model.py`

```bash
python -m transformers.convert_graph_to_onnx --framework pt --model bert-base-cased --quantize ./models/bert-base-cased.onnx
```

# Deploy

# Tests
