
# ONNX



**Resources**
* [transformers optimization documentation](https://huggingface.co/transformers/serialization.html?highlight=onnx)
* [Medium: Accelerate your NLP pipelines using Hugging Face Transformers and ONNX Runtime](https://medium.com/microsoftazure/accelerate-your-nlp-pipelines-using-hugging-face-transformers-and-onnx-runtime-2443578f4333)


Currently there is no version of `neuron-sdk` for Mac therefore I created a `dockerfile` to use it in a

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

# Tests
