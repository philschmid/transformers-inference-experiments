# Ray Serve: Scalable and Programmable Serving¶

Ray Serve is an easy-to-use scalable model serving library built on Ray. Ray Serve is:

* **Framework-agnostic:** Use a single toolkit to serve everything from deep learning models built with frameworks like PyTorch, Tensorflow, and Keras, to Scikit-Learn models, to arbitrary Python business logic.
* **Python-first:** Configure your model serving with pure Python code—no more YAML or JSON configs.

Since Ray Serve is built on Ray, it allows you to easily scale to many machines, both in your datacenter and in the cloud.

Ray Serve can be used in two primary ways to deploy your models at scale:

* Have Python functions and classes automatically placed behind HTTP endpoints.
* Alternatively, call them from within your [existing Python web server](https://docs.ray.io/en/master/serve/tutorials/web-server-integration.html#serve-web-server-integration-tutorial) using the Python-native [ServeHandle API](https://docs.ray.io/en/master/serve/package-ref.html#servehandle-api).

# Getting started

During these experiments, ray `v2.0.0` hasn't been release. Therefor we installed it from source.

```bash
pip install -r requirements.txt
```



# Resourcecs

* [Documentation](https://docs.ray.io/en/master/serve/index.html#rayserve)
* [Installation Documentation](https://docs.ray.io/en/master/installation.html)
* [Benchmarking](https://blog.einstein.ai/benchmarking-tensorrt-inference-server/)


# Current Example Implementations 

[fastAPI version with decoupled backend and ensemble prediction](./app.py)
