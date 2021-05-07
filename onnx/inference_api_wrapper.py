from transformers import AutoTokenizer, AutoModelForMaskedLM,pipeline
import numpy as np
import torch
from transformers.modeling_outputs import (  # CausalLMOutputWithPastAndCrossAttentions,
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
)
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions


def unwrap_onnx_inputs(kwargs, input_names):
    inputs = {}
    for onnxname in input_names:
        tmp_onnxname = onnxname
        if tmp_onnxname.startswith("output_"):
            L = len("output_")
            tmp_onnxname = tmp_onnxname[L:]

        if tmp_onnxname == "encoder_last_hidden_state":
            tmp_onnxname = "encoder_outputs.0"
        tokens = tmp_onnxname.split(".")

        d = kwargs
        try:
            for token in tokens:
                if token in d:
                    d = d[token]
                else:
                    token = int(token)
                    d = d[token]
        except Exception:
            raise Exception(f"{onnxname} not found in kwargs")

        input_ = d

        inputs[onnxname] = input_
    return inputs


def normalize_onnx_inputs(kwargs, input_names):
    inputs = unwrap_onnx_inputs(kwargs, input_names)
    return {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def normalize_onnx_outputs(outputs, output_names):
    outputs = [torch.from_numpy(o) for o in outputs]
    output_kwargs = wrap_onnx_outputs(outputs, output_names)
    return Seq2SeqLMOutput(**output_kwargs)


def wrap_onnx_outputs(outputs, output_names):
    assert len(outputs) == len(output_names), "Can't recreate outputs"
    output_kwargs = {}
    for output_name, output_value in zip(output_names, outputs):
        if output_name.startswith("output_"):
            L = len("output_")
            output_name = output_name[L:]

        tokens = output_name.split(".")
        d = output_kwargs
        previous_token = None
        previous_d = None
        for token in tokens:
            try:
                token = int(token)
            except Exception:
                pass

            if d is None:
                if isinstance(token, int):
                    previous_d[previous_token] = []
                    d = previous_d[previous_token]
                elif isinstance(token, str):
                    previous_d[previous_token] = {}
                    d = previous_d[previous_token]
                else:
                    raise Exception("Unexpected token")

            previous_d = d

            try:
                d = d[token]
            except KeyError:
                d[token] = None
                d = d[token]
            except IndexError:
                len_d = len(d)
                idx = token
                d += [None] * (idx + 1 - len_d)
                d = d[idx]
            previous_token = token
        previous_d[previous_token] = output_value

    return output_kwargs


def load_onnx_model_on_pipeline(nlp, quantized_output):
    sess_options = SessionOptions()

    # Set graph optimization level
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = InferenceSession(str(quantized_output), sess_options)

    model = nlp.model

    real_forward = model.forward

    input_names = [input_.name for input_ in sess.get_inputs()]
    output_names = [output_.name for output_ in sess.get_outputs()]

    def onnx_forward(*args, **kwargs):
        inputs = normalize_onnx_inputs(kwargs, input_names)
        outputs = sess.run(None, inputs)

        real_output = normalize_onnx_outputs(outputs, output_names)
        print("Successful inference on ONNX")
        return real_output

    def forward(*args, **kwargs):
        try:
            result = onnx_forward(*args, **kwargs)
            return result
        except Exception as e:
            print(f"Error running onnx_forward: {e}")
            return real_forward(*args, **kwargs)

    model.forward = forward