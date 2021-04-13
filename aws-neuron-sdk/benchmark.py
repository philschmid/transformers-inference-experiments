import tensorflow  # to workaround a protobuf version conflict issue
import torch
import torch.neuron
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

task = "mrpc"
split="validation"
all_datasets = load_dataset("glue", task)
metric = load_metric("glue", task)
dataset= all_datasets[split]

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
max_length=128
padding='max_length'

def preprocess_function(examples):
    # Tokenize the texts
    texts = (examples['sentence1'], examples['sentence2'])
    result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True,return_tensors="pt")
    result["labels"] = examples["label"]
    return result


def run_benchmark(raw_dataset,model,model_type):
    processed_dataset = raw_dataset.map(preprocess_function)
    processed_dataset = processed_dataset.select(range(1000))
    model_start = time.perf_counter()
#     model_type = 'neuron' if isinstance(model, torch.jit.ScriptModule) else 'torch'
    with torch.no_grad():
        for step, batch in enumerate(processed_dataset):
            input_ids = torch.tensor(batch['input_ids'])
            attention_mask = torch.tensor(batch['attention_mask'])
            token_type_ids = torch.tensor(batch['token_type_ids'])
            outputs = model(*[input_ids,attention_mask,token_type_ids])
            predictions = outputs[0][0].argmax().item()
            metric.add_batch(predictions=[predictions],references=[batch["labels"]])
        
    eval_metric = metric.compute()
    model_stop = time.perf_counter()
    total_time = round(model_stop - model_start,4)*1000
    average_time =  round(total_time/len(processed_dataset),4)
    return {'model_type':model_type,**eval_metric,'total_time':f"{total_time}ms",'average_time':f"{average_time}ms"}   


model_neuron = torch.jit.load('bert_neuron.pt')
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", return_dict=False)


model_res=run_benchmark(dataset, model,'pytorch')
model_neuron_res = run_benchmark(dataset, model_neuron,'neuron')

print(model_res)
print(model_neuron_res)