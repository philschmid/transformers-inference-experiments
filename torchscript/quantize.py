from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

max_length=256
model_id="bert-base-cased-finetuned-mrpc"
# Build tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, return_dict=False)

# Setup some example inputs
sequence_0 = "The company HuggingFace is based in New York City"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"


paraphrase = tokenizer.encode_plus(sequence_0, sequence_2, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")
example_inputs_paraphrase = paraphrase['input_ids'], paraphrase['attention_mask'], paraphrase['token_type_ids']
model(**paraphrase)[0]
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

traced_model = torch.jit.trace(quantized_model, example_inputs_paraphrase)
torch.jit.save(traced_model, "bert_traced_eager_quant.pt")