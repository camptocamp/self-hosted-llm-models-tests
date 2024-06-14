# self-hosted-llm-models-tests

## Notes

### GPU nodes

For running an LLM model on Kubernetes, we need nodes with a GPU installed.

On Exoscale, the more powerful GPUs available are the [GPU3 Instances](https://www.exoscale.com/pricing/#gpu3-instances). **These are only available on the `de-fra-1` region.**

Also, for every Exoscale subscription/organization, a support request needs to be made to allow the creation of these types of instances.

### Adding support for GPU workloads on Kubernetes

Exoscale provides a simple guide on how to enable GPU support in SKS nodes, available [here](https://community.exoscale.com/documentation/sks/gpu-sks-nodes/).

The Exoscale documentation note that the SKS cluster need to be on the `Pro` plan and not on the `Starter` plan, but I've been able to instantiate the GPU nodes on cluster with the latter.

From what I've gathered, the Exoscale GPU nodes already satisfy the [prerequisites](https://github.com/NVIDIA/k8s-device-plugin?tab=readme-ov-file#prerequisites) for the NVIDIA Device Plugin and we only need to install it. I choose to install it as an Helm chart and configure it through the values, as recommended for production deployments. More information about the NVIDIA utilities/plugins used below or on the README.md in the charts folder.

I did not explore all the [configuration possibilities](https://github.com/NVIDIA/k8s-device-plugin?tab=readme-ov-file#configuring-the-nvidia-device-plugin-binary) for the NVIDIA Device Plugin. Of notable interest seems the possibility of sharing a GPU through multiple workloads.
