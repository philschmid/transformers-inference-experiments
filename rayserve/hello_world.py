import requests

from ray import serve

ray.init()

serve.start(detached=True)


@serve.deployment
def hello(request):
    name = request.query_params["name"]
    return f"Hello {name}!"


hello.deploy()

# Query our endpoint over HTTP.
response = requests.get("http://127.0.0.1:8000/hello?name=serve").text
assert response == "Hello serve!"
