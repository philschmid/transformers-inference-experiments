import ray
from ray import serve
from typing import Optional

from fastapi import FastAPI, HTTPException
from transformers import pipeline
from pydantic import BaseModel

app = FastAPI()


class RequestBody(BaseModel):
    inputs: str
    parameters: Optional[dict] = None
    options: Optional[dict] = None


model_list = {
    "mobilebert": {"name": "mobilebert", "model_id": "lordtt13/emo-mobilebert", "init_replication": 1},
    "prunebert": {
        "name": "prunebert",
        "model_id": "huggingface/prunebert-base-uncased-6-finepruned-w-distil-mnli",
        "init_replication": 1,
    },
}


# Define our deployment.
def deploy_model(name, model_id):
    @serve.deployment(name=name, num_replicas=1)
    class Sentiment:
        def __init__(self, model_id):
            self.nlp_model = pipeline("sentiment-analysis", model=model_id)

        def __call__(self, request):
            return self.nlp_model(request)

    Sentiment.deploy(model_id)
    return Sentiment.get_handle()


@app.on_event("startup")  # Code to be run when the server starts.
async def startup_event():
    ray.init(address="auto")  # Connect to the running Ray cluster.
    serve.start(http_host=None)  # Start the Ray Serve instance.

    for model_config in model_list.values():
        model_config["handle"] = deploy_model(model_config["name"], model_config["model_id"])


@app.post("/{model}/predict")
async def sentiment(model: str, body: RequestBody):
    if model not in model_list.keys():
        raise HTTPException(status_code=404, detail=f"No Model {model} deployed")
    return await model_list[model]["handle"].remote(body.inputs)
