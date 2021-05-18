from tritonclient.utils import *
import tritonclient.http as httpclient
import time
import numpy as np

model_name = "huggingface_t5_translation"
example_input = "translate English to German: The green house is beautiful and big."

with httpclient.InferenceServerClient("localhost:8000") as client:
    input0_data = np.array([example_input], dtype=object)

    inputs = [
        httpclient.InferInput("INPUT0", input0_data.shape, "BYTES"),
    ]

    inputs[0].set_data_from_numpy(input0_data)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0", binary_data=True),
    ]
    start = time.time()
    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)
    end = time.time()

    result = response.get_response()
    print(f'translate: INPUT0 ({input0_data}) to OUTPUT0 ({response.as_numpy("OUTPUT0")[0].decode("utf-8")})')
    print(f"request took: {round((end-start)*1000,0)}ms")
