from os import name
import ray
from ray import serve
from typing import Optional

from fastapi import FastAPI, HTTPException
from ray.state import available_resources
from starlette.responses import JSONResponse
from transformers import pipeline
from pydantic import BaseModel

app = FastAPI()


class RequestBody(BaseModel):
    inputs: str
    parameters: Optional[dict] = None
    options: Optional[dict] = None


model_list = {
    "mobilebert": {
        "name": "mobilebert",
        "task": "sentiment-analysis",
        "model_id": "lordtt13/emo-mobilebert",
        "init_replication": 1,
    },
    "prunebert": {
        "name": "prunebert",
        "model_id": "huggingface/prunebert-base-uncased-6-finepruned-w-distil-mnli",
        "init_replication": 1,
    },
}


# Define our deployment.
def deploy_model(name, model_id, task):
    @serve.deployment(name=name, num_replicas=1)
    class Transformer:
        def __init__(self, model_id, task):
            self.nlp_model = pipeline(task, model=model_id)

        def __call__(self, request):
            return self.nlp_model(request)

    Transformer.deploy(model_id, task)
    return Transformer.get_handle()


@app.on_event("startup")  # Code to be run when the server starts.
async def startup_event():
    ray.init(address="auto")  # Connect to the running Ray cluster.
    serve.start(http_host=None)  # Start the Ray Serve instance.

    for model_config in model_list.values():
        model_config["handle"] = deploy_model(
            name=model_config["name"], model_id=model_config["model_id"], task=model_config["task"]
        )


@app.post("/{model}/predict")
async def sentiment(model: str, body: RequestBody):
    if model not in model_list.keys():
        raise HTTPException(status_code=404, detail=f"No Model {model} deployed")
    return await model_list[model]["handle"].remote(body.inputs)


@app.post("/{model}/scale/{to}")
async def scale_up_or_down(model: str, to: int):
    if model not in model_list.keys():
        raise HTTPException(status_code=404, detail=f"No Model {model} deployed")
    available_resources = ray.available_resources()
    if to > available_resources["CPU"]:
        raise HTTPException(
            status_code=404,
            detail=f"Scaling model {model} to {to} not possible only {available_resources['CPU']} left",
        )
    deployment = serve.get_deployment(model)
    deployment.options(num_replicas=to).deploy()
    return serve.get_backend_config(model).__dict__


@app.get("/backends")
async def list_backends():
    return serve.list_backends()


@app.get("/context")
async def context():
    return {"total_resources": ray.cluster_resources(), "available_resources": ray.available_resources()}


@app.get("/endpoints")
async def list_endpoints():
    return serve.list_endpoints()


class DeploymentBody(BaseModel):
    name: str
    task: str
    model_id: str


@app.post("/deploy")
async def deploy_new_model(model: DeploymentBody):
    available_resources = ray.available_resources()
    if available_resources["CPU"] == 0:
        raise HTTPException(status_code=404, detail=f"No Resources for deployed")
    model_list[model.name] = model.__dict__
    model_list[model.name]["handle"] = deploy_model(name=model.name, model_id=model.model_id, task=model.task)
    return "Deployment Successfull"
