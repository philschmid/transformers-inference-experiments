# KFServing

KFServing abstracts away the complexity of server configuration, networking, health checking, autoscaling of heterogeneous hardware (CPU, GPU, TPU), scaling from zero, and progressive (aka. canary) rollouts. It provides a complete story for production ML serving that includes prediction, pre-processing, post-processing and explainability, in a way that is compatible with various frameworks – Tensorflow, PyTorch, XGBoost, ScikitLearn, and ONNX.

Check out the KFServing [repository on github](https://github.com/kubeflow/kfserving) for a deeper dive.

Knative is a Kubernetes-based platform to deploy and manage modern serverless workloads. This grants to KFServing the properties of:

- Scaling to and from zero: optimizing costs associated with inference
- Auto-scale of GPUs and TPUs: reducing latency with specialized hardware at a cost per demand

Istio is a service mesh technology that works through the concept of Kubernetes sidecars. A sidecar container is added to every pod, handling all network traffic. This enables:

- Canary roll-outs: allowing for safe model updates across users
- Traffic to the model: routing and ingress management
- Observability: tracing, monitoring, and logging features for your models
- Load balancing: HTTP, gRPC, WebSocket, and TCP traffic
- Security: authentication, authorization, and encryption of service communication at scale
- KFServing itself provides a Kubernetes Custom Resource Definition – an object that extends the Kubernetes API – specific for serving machine learning models saved on various frameworks, such as Tensorflow, PyTorch, XGBoost, ScikitLearn, and ONNX, into production environments.


# Resources

* [Documentation](https://www.kubeflow.org/docs/components/kfserving/kfserving/)
* [Github](https://github.com/kubeflow/kfserving)
* [AWS Deployment](https://www.kubeflow.org/docs/distributions/aws/customizing-aws/customizing-aws/)
* [KFServing Server](https://github.com/kubeflow/kfserving/tree/master/python/kfserving)
* [Deploying Huggingface with kfserving](http://www.pattersonconsultingtn.com/blog/deploying_huggingface_with_kfserving.html)
* [Service yaml for custom predictor v1beta1](https://github.com/kubeflow/kfserving/blob/master/docs/samples/v1beta1/custom/simple.yaml)
* [Multiple custom predictor yamls using torchserve](https://github.com/kubeflow/kfserving/tree/master/docs/samples/v1beta1/custom/torchserve)
* [KFServer Infernece Toolkit](https://github.com/kubeflow/kfserving/blob/master/python/kfserving/kfserving/kfmodel.py)


# Setting Up KFServing

## Install `minikube` and start k8s

To demo the Hugging Face model on KFServing we'll use the local quick install method on a [minikube kubernetes cluster](https://kubernetes.io/docs/tasks/tools/install-minikube/). 

To install KFServing standalone on minikube we will first need to install the dependencies:

* kustomize v3.5.4+
* kubectl
* helm 3

```bash
brew install minikube
brew install kustomize
brew install helm
```

After `minikube` is running adjust resources for it

```bash
minikube config set memory 6500
minikube config set cpus 4
```

Now, you can start `minikube` with 
```bash
minikube start
minikube status
```

## Install KFServing on `minikube`

KFserving is providing a “quick install” script, which installs Istio, KNative, KFserving for us without having to install all of Kubeflow and the extra components that tend to slow down local demo installs.

```bash
./intall_kfserving.sh

kubectl get po -n kfserving-system
```

## Deploy a container to KFServing

In `examples/` is an example using `KFServing` (Python KFserver) with `transformers` pipeline. You can build the container with

```bash
make build
```

you can test the container locally with
```bash
make start
```
And then request `localhost:8080/v2/models/kfserving-custom-model/infer` with

```json
{
	"inputs": [
		"I am really happy how kfserving is working"
	],
	"parameters": {
		"test": 1
	}
}
```

Afterwards deploy it to `kfserving` with.

```bash
kubectl apply -f example/service.yaml
```
Once we run the above kubectl command, we should have a working InferenceService running on our local kubernetes cluster. We can check the status of our model with the kubectl command:

```bash
kubectl get inferenceservices
```

## Making an Inference Call
First we need to do some port forwarding work so our model's port is exposed to our local system with the command:

```bash
kubectl port-forward --namespace istio-system $(kubectl get pod --namespace istio-system --selector="app=istio-ingressgateway" --output jsonpath='{.items[0].metadata.name}') 8080:8080
```
Request with `curl`

```bash
curl --request POST \
  --url http://localhost:8080/v2/models/transformers-custom-model/infer \
  --header 'Content-Type: application/json' \
  --header 'Host: transformers-predictor-default.default.example.com' \
  --data '{
	"inputs": [
		"I am really happy how kfserving is working"
	],
	"parameters": {
		"test": 1
	}
}'
```



## KFServing Routes

**GET**
`/v2/models` -> get models
`/v2/models/<model name>/status` ->  Model Health API returns 200 if model is ready to serve
**POST**
`/v2/models/<model name>/infer` -> call `.predict()` function
`/v2/repository/models/<model name>/load` -> load model
`/v2/repository/models/<model name>/unload` -> unload model

