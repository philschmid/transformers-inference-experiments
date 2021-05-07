.PHONY: build

tag=ubuntu-onnx
cwd  := $(shell pwd)

build:
	docker build --tag ${tag} \
							 --file ./Dockerfile \
							 .

start:
	docker run -t -i -p 8888:8888 -v ${cwd}:/home/onnx ${tag} 