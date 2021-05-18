# triton `python_backend` example using `transformers` model 

The Triton backend for Python. The goal of Python backend is to let you serve models written in Python by Triton Inference Server without having to write any C++ code.


## Result 

```bash
python client.py 
translate: INPUT0 (['translate English to German: The green house is beautiful and big.']) to OUTPUT0 (Das grüne Haus ist schön und groß.)
request took: 181.0ms
```

=>  native python inference with pipelines, without http takes ~200ms -> 10% performance increase using triton

## Getting started 

1. build trition-container with

```bash
make build
```

2. build triton-client-container with

```bash
make build-client
```

3. start the triton server with 
```bash
make start
```

3a. start interactive session in triton container
```bash
make run
```


4. run the `client.py` against the inference server with
```bash
make test
```

4a. if you want to have an interactive test client run
```bash
make test-i
```
