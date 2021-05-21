import kfserving
import logging
import os
from typing import Dict
from transformers import pipeline

logger = logging.getLogger(__name__)


class KFServingSampleModel(kfserving.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.task = os.environ.get("TASK", None)
        self.model_id = os.environ.get("MODEL_ID", None)
        self.ready = False

    def load(self):
        logger.info("Loading the model")
        self.pipeline = pipeline(self.task, model=self.model_id)
        self.ready = True

    def predict(self, data: Dict) -> Dict:
        logger.info(data)
        # pop inputs for pipeline
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", None)

        # start_time = time.time()
        # pass inputs with all kwargs in data
        if parameters is not None:
            prediction = self.pipeline(inputs, **parameters)
        else:
            prediction = self.pipeline(inputs)
        # logger.info(f"inference time {time.time()-start_time}s")
        return {"pred": prediction}


if __name__ == "__main__":
    model = KFServingSampleModel("transformers-custom-model")
    model.load()
    kfserving.KFServer(workers=1).start([model])
