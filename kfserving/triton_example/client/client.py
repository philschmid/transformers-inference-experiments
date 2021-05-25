import kfserving
import logging
import os
from typing import Dict
from transformers import AutoTokenizer
import tritonclient.http as httpclient
import numpy as np

logger = logging.getLogger(__name__)


class KFServingSampleModel(kfserving.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.model_id = os.environ.get("MODEL_ID", None)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.triton_client = None
        self.ignore_labels = ["O"]
        self.id2label = {
            0: "O",
            1: "B-PER",
            2: "I-PER",
            3: "B-ORG",
            4: "I-ORG",
            5: "B-LOC",
            6: "I-LOC",
            7: "B-MISC",
            8: "I-MISC",
        }

    # def load(self):
    #     logger.info("Loading the model")
    #     self.pipeline = pipeline(self.task, model=self.model_id)
    #     self.ready = True

    def preprocess(self, data: str) -> Dict:
        logger.info(data)
        # pop inputs for pipeline
        self.inputs = data.pop("inputs", data)
        tokens = self.tokenizer(
            self.inputs,
            return_attention_mask=False,
            return_tensors="np",
            truncation=True,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
        )

        self.offset_mapping = tokens.pop("offset_mapping")
        self.special_tokens_mask = tokens.pop("special_tokens_mask")
        self.input_ids = self.input_ids

        return tokens

    def predict(self, data: Dict) -> Dict:
        logger.info(data)
        if not self.triton_client:
            self.triton_client = httpclient.InferenceServerClient(url=self.predictor_host, verbose=True)

        inputs = [
            httpclient.InferInput("input_ids", [1, 128], "INT32"),
        ]
        inputs[0].set_data_from_numpy(data["input_ids"])

        outputs = [
            httpclient.InferRequestedOutput("logits", binary_data=False),
        ]
        result = self.triton_client.infer(self.model_id, inputs, outputs=outputs)
        return result.get_response()

    def postprocess(self, result) -> Dict:
        logger.info(result)

        logger.info(result["outputs"][0]["data"])

        entities = result["outputs"][0]["data"][0]

        score = np.exp(entities) / np.exp(entities).sum(-1, keepdims=True)
        labels_idx = score.argmax(axis=-1)
        entities = []
        filtered_labels_idx = []

        for idx, label_idx in enumerate(labels_idx):
            if (
                self.id2label[label_idx] not in self.ignore_labels
                and self.id2label[label_idx] is not self.special_tokens_mask[idx]
            ):
                filtered_labels_idx.append((idx, label_idx))

            for idx, label_idx in filtered_labels_idx:
                start_ind, end_ind = self.offset_mapping[idx]
                word_ref = self.inputs[start_ind:end_ind]
                word = self.tokenizer.convert_ids_to_tokens(int(self.input_ids[idx]))
                is_subword = len(word_ref) != len(word)

                if int(self.input_ids[idx]) == self.tokenizer.unk_token_id:
                    word = word_ref
                entity = {
                    "word": word,
                    "score": score[idx][label_idx].item(),
                    "entity": self.id2label[label_idx],
                    "index": idx,
                    "start": start_ind,
                    "end": end_ind,
                }

                entities += [entity]
            return {"predictions": entities}


if __name__ == "__main__":
    model = KFServingSampleModel("transformers-custom-model")
    model.load()
    kfserving.KFServer(workers=1).start([model])
