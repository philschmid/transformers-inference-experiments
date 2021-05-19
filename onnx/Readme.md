
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


**Open Research links**
- https://github.com/Ki6an/fastT5/tree/8dda859086af631a10ad210a5f1afdec64d49616
- https://stackoverflow.com/questions/66109084/how-to-convert-huggingfaces-seq2seq-models-to-onnx-format/66117248#66117248
- http://www.pattersonconsultingtn.com/blog/deploying_huggingface_with_kfserving.html
- https://developer.nvidia.com/blog/announcing-onnx-runtime-for-jetson/
- https://s3.us-east.cloud-object-storage.appdomain.cloud/staging-sombra/default/series/os-kubeflow-2020/static/kubeflow04.pdf
- https://www.programcreek.com/python/example/126284/transformers.BertTokenizer.from_pretrained 
- https://opendatascience.com/accelerate-your-nlp-pipelines-using-hugging-face-transformers-and-onnx-runtime/
- https://opendatascience.com/ml-inference-on-edge-devices-with-onnx-runtime-using-azure-devops/
- https://www.kaggle.com/alexalex02/nlp-transformers-inference-optimization/code
- https://simpletransformers.ai/docs/tips-and-tricks/#onnx-support-beta
- https://cloudblogs.microsoft.com/opensource/2021/03/01/optimizing-bert-model-for-intel-cpu-cores-using-onnx-runtime-default-execution-provider/

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
