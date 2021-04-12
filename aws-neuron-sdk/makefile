.PHONY: build

tag=amazonlinux2-neuron
cwd  := $(shell pwd)

build:
	docker build --tag ${tag} \
							 --file ./Dockerfile \
							 .

start:
	docker run -t -i -p 8888:8888 -v ${cwd}:/home/neuron ${tag} 