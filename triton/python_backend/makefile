.PHONY: build build-client start run test test-i
backend=python
tag=triton-${backend}
cwd := $(shell pwd)
model=huggingface_t5_translation


build:
	docker build --tag ${tag} \
							 --file ./dockerfile \
							 .

build-client:
	docker build --tag ${tag}-client \
							 --file ./dockerfile.client \
							 .

start:
	docker run 	--shm-size=1g \
		--ulimit memlock=-1 \
		-p 8000:8000 \
		-p 8001:8001 \
		-p 8002:8002 \
		-v ${cwd}/models/${model}:/opt/tritonserver/models/${model} \
		--ulimit stack=67108864 \
		-ti \
		${tag}

run:
	docker run --shm-size=1g \
		--ulimit memlock=-1 \
		-p 8000:8000 \
		-p 8001:8001 \
		-p 8002:8002 \
		-v ${cwd}/models/${model}:/opt/tritonserver/models/${model} \
		--ulimit stack=67108864 \
		-ti \
		${tag} \
		bash

test:
	make build-client && docker run -ti --net host ${tag}-client

test-i:
	make build-client && docker run -ti -v ${cwd}/client.py:/client.py  --net host ${tag}-client bash