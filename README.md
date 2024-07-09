# self-hosted-llm-models-tests

## Notes

### GPU nodes

For running an LLM model on Kubernetes, we need nodes with a GPU installed.

On Exoscale, the more powerful GPUs available are the [GPU3 Instances](https://www.exoscale.com/pricing/#gpu3-instances). **These are only available on the `de-fra-1` region.**

Also, for every Exoscale subscription/organization, a support request needs to be made to allow the creation of these types of instances.

### Adding support for GPU workloads on Kubernetes

Exoscale provides a simple guide on how to enable GPU support in SKS nodes, available [here](https://community.exoscale.com/documentation/sks/gpu-sks-nodes/).

The Exoscale documentation note that the SKS cluster needs to be on the `Pro` plan and not on the `Starter` plan, but I've been able to instantiate the GPU nodes on a cluster with the latter.

From what I've gathered, the Exoscale GPU nodes already satisfy the [prerequisites](https://github.com/NVIDIA/k8s-device-plugin?tab=readme-ov-file#prerequisites) for the NVIDIA Device Plugin and we only need to install it. I choose to install it as an Helm chart and configure it through the values, as recommended for production deployments. More information about the NVIDIA utilities or plugins used will the [README.md in the charts folder](./charts/README.md).

I did not explore all the [configuration possibilities](https://github.com/NVIDIA/k8s-device-plugin?tab=readme-ov-file#configuring-the-nvidia-device-plugin-binary) for the NVIDIA Device Plugin. Of notable interest seems the possibility of sharing a GPU through multiple workloads (by default, if a pod needs a GPU, it will be exclusively attached to it a not any other pod).

### Running LLM models using Hugging Face's Text Generation Inference

I've created a simple chart [here](./charts/apps/text-generation-inference/) that deploys Hugging Face's Text Generation Inference on a Kubernetes cluster. The [`values.yaml`](./charts/apps/text-generation-inference/values.yaml) is already configured to schedule the deployment on the GPU nodepool created by [this Terraform](./terraform/main.tf) code.

I've also added an ApplicationSet to automatically deploy the NVIDIA Device Plugin and the Text Generation Inference deployment, which is also added by Terraform [here](./terraform/apps.tf). However, to access the model, note that you need to add a Kubernetes secret containing your Hugging Face token to give you access to the model. You can use a command similar to the following (**adapt it depending on your deployment and local environment**):

```shell
kubectl --kubeconfig ~/.kube/is-sandbox-exo-gh-llm-sks-cluster.config -n huggingface-apps create secret generic huggingface-token --from-literal token=YOUR-TOKEN-HERE
```

### Tested models

#### meta-llama/Meta-Llama-3-8B-Instruct

Using TGI, running this model was quite easy. I've tested its functionality by port-forwarding directly to the pod created by [this chart](./charts/apps/text-generation-inference) and then running a command like the following:

```shell
curl 127.0.0.1:8080/generate_stream \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":200}}' \
    -H 'Content-Type: application/json'
```

**I've not customized the prompt parameters any further than increasing and decreasing the `max_new_tokens` parameter.**

#### meta-llama/Meta-Llama-3-70B-Instruct

Even with the GPU3 medium instances from Exoscale, I'm unable to run this model. From what I've gathered, the limiting factor seems to be the GPU memory which is not enough. I did not find a way to maybe decrease the memory requirements by decreasing the precision of the model...

For reference, here are the logs from the pod:

```
stream logs failed container "text-generation-inference" in pod "text-generation-inference-67869c9b65-6j2px" is waiting to start: ContainerCreating for huggingface-apps/text-generation-inference-67869c9b65-6j2px (text-generation-inference)
stream logs failed container "text-generation-inference" in pod "text-generation-inference-67869c9b65-6j2px" is waiting to start: ContainerCreating for huggingface-apps/text-generation-inference-67869c9b65-6j2px (text-generation-inference)
stream logs failed container "text-generation-inference" in pod "text-generation-inference-67869c9b65-6j2px" is waiting to start: ContainerCreating for huggingface-apps/text-generation-inference-67869c9b65-6j2px (text-generation-inference)
stream logs failed container "text-generation-inference" in pod "text-generation-inference-67869c9b65-6j2px" is waiting to start: ContainerCreating for huggingface-apps/text-generation-inference-67869c9b65-6j2px (text-generation-inference)
stream logs failed container "text-generation-inference" in pod "text-generation-inference-67869c9b65-6j2px" is waiting to start: ContainerCreating for huggingface-apps/text-generation-inference-67869c9b65-6j2px (text-generation-inference)
stream logs failed container "text-generation-inference" in pod "text-generation-inference-67869c9b65-6j2px" is waiting to start: ContainerCreating for huggingface-apps/text-generation-inference-67869c9b65-6j2px (text-generation-inference)
2024-06-17T13:11:09.417238Z  INFO text_generation_launcher: Args {
    model_id: "meta-llama/Meta-Llama-3-70B-Instruct",
    revision: None,
    validation_workers: 2,
    sharded: None,
    num_shard: None,
    quantize: None,
    speculate: None,
    dtype: None,
    trust_remote_code: false,
    max_concurrent_requests: 128,
    max_best_of: 2,
    max_stop_sequences: 4,
    max_top_n_tokens: 5,
    max_input_tokens: None,
    max_input_length: None,
    max_total_tokens: None,
    waiting_served_ratio: 0.3,
    max_batch_prefill_tokens: None,
    max_batch_total_tokens: None,
    max_waiting_tokens: 20,
    max_batch_size: None,
    cuda_graphs: None,
    hostname: "text-generation-inference-67869c9b65-6j2px",
    port: 8080,
    shard_uds_path: "/tmp/text-generation-server",
    master_addr: "localhost",
    master_port: 29500,
    huggingface_hub_cache: Some(
        "/data",
    ),
    weights_cache_override: None,
    disable_custom_kernels: false,
    cuda_memory_fraction: 1.0,
    rope_scaling: None,
    rope_factor: None,
    json_output: false,
    otlp_endpoint: None,
    cors_allow_origin: ],
    watermark_gamma: None,
    watermark_delta: None,
    ngrok: false,
    ngrok_authtoken: None,
    ngrok_edge: None,
    tokenizer_config_path: None,
    disable_grammar_support: false,
    env: false,
    max_client_batch_size: 4,
}
2024-06-17T13:11:09.417323Z  INFO hf_hub: Token file not found "/root/.cache/huggingface/token"    
2024-06-17T13:11:09.543606Z  INFO text_generation_launcher: Default `max_input_tokens` to 4095
2024-06-17T13:11:09.543628Z  INFO text_generation_launcher: Default `max_total_tokens` to 4096
2024-06-17T13:11:09.543631Z  INFO text_generation_launcher: Default `max_batch_prefill_tokens` to 4145
2024-06-17T13:11:09.543633Z  INFO text_generation_launcher: Using default cuda graphs [1, 2, 4, 8, 16, 32]
2024-06-17T13:11:09.543642Z  INFO text_generation_launcher: Sharding model on 2 processes
2024-06-17T13:11:09.543717Z  INFO download: text_generation_launcher: Starting download process.
2024-06-17T13:11:12.946148Z  INFO text_generation_launcher: Files are already present on the host. Skipping download.
2024-06-17T13:11:13.648486Z  INFO download: text_generation_launcher: Successfully downloaded weights.
2024-06-17T13:11:13.648865Z  INFO shard-manager: text_generation_launcher: Starting shard rank=0
2024-06-17T13:11:13.648935Z  INFO shard-manager: text_generation_launcher: Starting shard rank=1
2024-06-17T13:11:16.975709Z  INFO text_generation_launcher: Detected system cuda
2024-06-17T13:11:17.016998Z  INFO text_generation_launcher: Detected system cuda
2024-06-17T13:11:23.660367Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:11:23.660559Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:11:33.670536Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:11:33.670835Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:11:43.680628Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:11:43.680705Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:11:53.689770Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:11:53.689773Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:12:03.698378Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:12:03.699114Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:12:13.707648Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:12:13.708747Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:12:23.717162Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:12:23.718950Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:12:33.728045Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:12:33.728556Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:12:43.737759Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:12:43.738366Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:12:53.747832Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:12:53.748218Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:13:03.756532Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:13:03.756944Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:13:13.766353Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:13:13.767390Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:13:23.775729Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:13:23.776327Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:13:33.785201Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:13:33.785244Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:13:43.794274Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:13:43.798835Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:13:53.803290Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:13:53.807702Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:14:03.812941Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:14:03.817774Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:14:13.822419Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:14:13.828131Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:14:23.832744Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:14:23.838818Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:14:33.843034Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:14:33.848408Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:14:43.853939Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:14:43.858796Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:14:53.863016Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:14:53.867611Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:15:03.873177Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:15:03.877365Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:15:13.882463Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:15:13.886028Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:15:23.893656Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:15:23.895302Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:15:33.905075Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:15:33.905375Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:15:43.914435Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:15:43.915630Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:15:53.924147Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:15:53.926449Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:16:03.933869Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:16:03.936361Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:16:13.942833Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:16:13.945206Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:16:23.952885Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:16:23.954552Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:16:33.962415Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:16:33.963775Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:16:43.971685Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:16:43.973422Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:16:53.982154Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:16:53.982169Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:17:03.991395Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:17:03.991561Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:17:14.001024Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:17:14.001634Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:17:24.010221Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:17:24.010512Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:17:34.019684Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:17:34.019699Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:17:44.028716Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:17:44.029137Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:17:54.037855Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:17:54.038852Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:18:04.047265Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:18:04.047798Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:18:14.056664Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:18:14.058514Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-06-17T13:18:24.066369Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-06-17T13:18:24.067545Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/server.py", line 263, in serve
    asyncio.run(
  File "/opt/conda/lib/python3.10/asyncio/runners.py", line 44, in run
    return loop.run_until_complete(main)
  File "/opt/conda/lib/python3.10/asyncio/base_events.py", line 636, in run_until_complete
    self.run_forever()
  File "/opt/conda/lib/python3.10/asyncio/base_events.py", line 603, in run_forever
    self._run_once()
  File "/opt/conda/lib/python3.10/asyncio/base_events.py", line 1909, in _run_once
    handle._run()
  File "/opt/conda/lib/python3.10/asyncio/events.py", line 80, in _run
    self._context.run(self._callback, *self._args)
> File "/opt/conda/lib/python3.10/site-packages/text_generation_server/server.py", line 225, in serve_inner
    model = get_model(
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/models/__init__.py", line 591, in get_model
    return FlashLlama(
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/models/flash_llama.py", line 74, in __init__
    model = FlashLlamaForCausalLM(prefix, config, weights)
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/models/custom_modeling/flash_llama_modeling.py", line 402, in __init__
    self.model = FlashLlamaModel(prefix, config, weights)
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/models/custom_modeling/flash_llama_modeling.py", line 326, in __init__
    [
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/models/custom_modeling/flash_llama_modeling.py", line 327, in <listcomp>
    FlashLlamaLayer(
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/models/custom_modeling/flash_llama_modeling.py", line 269, in __init__
    self.mlp = LlamaMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/models/custom_modeling/flash_llama_modeling.py", line 222, in __init__
    self.gate_up_proj = TensorParallelColumnLinear.load_multi(
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/layers/tensor_parallel.py", line 175, in load_multi
    weight = weights.get_multi_weights_col(
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/utils/weights.py", line 439, in get_multi_weights_col
    w = [self.get_sharded(f"{p}.weight", dim=0) for p in prefixes]
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/utils/weights.py", line 439, in <listcomp>
    w = [self.get_sharded(f"{p}.weight", dim=0) for p in prefixes]
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/utils/weights.py", line 131, in get_sharded
    return self.get_partial_sharded(tensor_name, dim)
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/utils/weights.py", line 119, in get_partial_sharded
    tensor = tensor.to(device=self.device)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 224.00 MiB. GPU
2024-06-17T13:18:26.673888Z ERROR text_generation_launcher: Error when initializing model
Traceback (most recent call last):
  File "/opt/conda/bin/text-generation-server", line 8, in <module>
    sys.exit(app())
  File "/opt/conda/lib/python3.10/site-packages/typer/main.py", line 311, in __call__
    return get_command(self)(*args, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/click/core.py", line 1157, in __call__
    return self.main(*args, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/typer/core.py", line 778, in main
    return _main(
  File "/opt/conda/lib/python3.10/site-packages/typer/core.py", line 216, in _main
    rv = self.invoke(ctx)
  File "/opt/conda/lib/python3.10/site-packages/click/core.py", line 1688, in invoke
    return _process_result(sub_ctx.command.invoke(sub_ctx))
  File "/opt/conda/lib/python3.10/site-packages/click/core.py", line 1434, in invoke
    return ctx.invoke(self.callback, **ctx.params)
  File "/opt/conda/lib/python3.10/site-packages/click/core.py", line 783, in invoke
    return __callback(*args, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/typer/main.py", line 683, in wrapper
    return callback(**use_params)  # type: ignore
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/cli.py", line 93, in serve
    server.serve(
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/server.py", line 263, in serve
    asyncio.run(
  File "/opt/conda/lib/python3.10/asyncio/runners.py", line 44, in run
    return loop.run_until_complete(main)
  File "/opt/conda/lib/python3.10/asyncio/base_events.py", line 636, in run_until_complete
    self.run_forever()
  File "/opt/conda/lib/python3.10/asyncio/base_events.py", line 603, in run_forever
    self._run_once()
  File "/opt/conda/lib/python3.10/asyncio/base_events.py", line 1909, in _run_once
    handle._run()
  File "/opt/conda/lib/python3.10/asyncio/events.py", line 80, in _run
    self._context.run(self._callback, *self._args)
> File "/opt/conda/lib/python3.10/site-packages/text_generation_server/server.py", line 225, in serve_inner
    model = get_model(
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/models/__init__.py", line 591, in get_model
    return FlashLlama(
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/models/flash_llama.py", line 74, in __init__
    model = FlashLlamaForCausalLM(prefix, config, weights)
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/models/custom_modeling/flash_llama_modeling.py", line 402, in __init__
    self.model = FlashLlamaModel(prefix, config, weights)
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/models/custom_modeling/flash_llama_modeling.py", line 326, in __init__
    [
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/models/custom_modeling/flash_llama_modeling.py", line 327, in <listcomp>
    FlashLlamaLayer(
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/models/custom_modeling/flash_llama_modeling.py", line 269, in __init__
    self.mlp = LlamaMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/models/custom_modeling/flash_llama_modeling.py", line 222, in __init__
    self.gate_up_proj = TensorParallelColumnLinear.load_multi(
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/layers/tensor_parallel.py", line 175, in load_multi
    weight = weights.get_multi_weights_col(
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/utils/weights.py", line 439, in get_multi_weights_col
    w = [self.get_sharded(f"{p}.weight", dim=0) for p in prefixes]
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/utils/weights.py", line 439, in <listcomp>
    w = [self.get_sharded(f"{p}.weight", dim=0) for p in prefixes]
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/utils/weights.py", line 131, in get_sharded
    return self.get_partial_sharded(tensor_name, dim)
  File "/opt/conda/lib/python3.10/site-packages/text_generation_server/utils/weights.py", line 119, in get_partial_sharded
    tensor = tensor.to(device=self.device)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 224.00 MiB. GPU  has a total capacity of 44.34 GiB of which 160.81 MiB is free. Process 47694 has 44.17 GiB memory in use. Of the allocated memory 43.36 GiB is allocated by PyTorch, and 396.27 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
2024-06-17T13:18:29.371622Z ERROR shard-manager: text_generation_launcher: Shard complete standard error output:

/opt/conda/lib/python3.10/site-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
[rank1]: Traceback (most recent call last):

[rank1]:   File "/opt/conda/bin/text-generation-server", line 8, in <module>
[rank1]:     sys.exit(app())

[rank1]:   File "/opt/conda/lib/python3.10/site-packages/text_generation_server/cli.py", line 93, in serve
[rank1]:     server.serve(

[rank1]:   File "/opt/conda/lib/python3.10/site-packages/text_generation_server/server.py", line 263, in serve
[rank1]:     asyncio.run(

[rank1]:   File "/opt/conda/lib/python3.10/asyncio/runners.py", line 44, in run
[rank1]:     return loop.run_until_complete(main)

[rank1]:   File "/opt/conda/lib/python3.10/asyncio/base_events.py", line 649, in run_until_complete
[rank1]:     return future.result()

[rank1]:   File "/opt/conda/lib/python3.10/site-packages/text_generation_server/server.py", line 225, in serve_inner
[rank1]:     model = get_model(

[rank1]:   File "/opt/conda/lib/python3.10/site-packages/text_generation_server/models/__init__.py", line 591, in get_model
[rank1]:     return FlashLlama(

[rank1]:   File "/opt/conda/lib/python3.10/site-packages/text_generation_server/models/flash_llama.py", line 74, in __init__
[rank1]:     model = FlashLlamaForCausalLM(prefix, config, weights)

[rank1]:   File "/opt/conda/lib/python3.10/site-packages/text_generation_server/models/custom_modeling/flash_llama_modeling.py", line 402, in __init__
[rank1]:     self.model = FlashLlamaModel(prefix, config, weights)

[rank1]:   File "/opt/conda/lib/python3.10/site-packages/text_generation_server/models/custom_modeling/flash_llama_modeling.py", line 326, in __init__
[rank1]:     [

[rank1]:   File "/opt/conda/lib/python3.10/site-packages/text_generation_server/models/custom_modeling/flash_llama_modeling.py", line 327, in <listcomp>
[rank1]:     FlashLlamaLayer(

[rank1]:   File "/opt/conda/lib/python3.10/site-packages/text_generation_server/models/custom_modeling/flash_llama_modeling.py", line 269, in __init__
[rank1]:     self.mlp = LlamaMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)

[rank1]:   File "/opt/conda/lib/python3.10/site-packages/text_generation_server/models/custom_modeling/flash_llama_modeling.py", line 222, in __init__
[rank1]:     self.gate_up_proj = TensorParallelColumnLinear.load_multi(

[rank1]:   File "/opt/conda/lib/python3.10/site-packages/text_generation_server/layers/tensor_parallel.py", line 175, in load_multi
[rank1]:     weight = weights.get_multi_weights_col(

[rank1]:   File "/opt/conda/lib/python3.10/site-packages/text_generation_server/utils/weights.py", line 439, in get_multi_weights_col
[rank1]:     w = [self.get_sharded(f"{p}.weight", dim=0) for p in prefixes]

[rank1]:   File "/opt/conda/lib/python3.10/site-packages/text_generation_server/utils/weights.py", line 439, in <listcomp>
[rank1]:     w = [self.get_sharded(f"{p}.weight", dim=0) for p in prefixes]

[rank1]:   File "/opt/conda/lib/python3.10/site-packages/text_generation_server/utils/weights.py", line 131, in get_sharded
[rank1]:     return self.get_partial_sharded(tensor_name, dim)

[rank1]:   File "/opt/conda/lib/python3.10/site-packages/text_generation_server/utils/weights.py", line 119, in get_partial_sharded
[rank1]:     tensor = tensor.to(device=self.device)

[rank1]: torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 224.00 MiB. GPU  has a total capacity of 44.34 GiB of which 160.81 MiB is free. Process 47694 has 44.17 GiB memory in use. Of the allocated memory 43.36 GiB is allocated by PyTorch, and 396.27 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
 rank=1
2024-06-17T13:18:29.406790Z ERROR text_generation_launcher: Shard 1 failed to start
2024-06-17T13:18:29.406822Z  INFO text_generation_launcher: Shutting down shards
2024-06-17T13:18:29.472875Z  INFO shard-manager: text_generation_launcher: Terminating shard rank=0
2024-06-17T13:18:29.473196Z  INFO shard-manager: text_generation_launcher: Waiting for shard to gracefully shutdown rank=0
2024-06-17T13:18:30.174163Z  INFO shard-manager: text_generation_launcher: shard terminated rank=0
Error: ShardCannotStart
Stream closed EOF for huggingface-apps/text-generation-inference-67869c9b65-6j2px (text-generation-inference)
```

### Running models with [llama.cpp](https://github.com/ggerganov/llama.cpp/tree/master)

#### Interacting with the [API](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#api-endpoints)

```bash
curl http://localhost:8080/completion \
    -d '{"prompt":"What is Deep Learning?", "n_predict": 128}' \
    -H 'Content-Type: application/json'
```