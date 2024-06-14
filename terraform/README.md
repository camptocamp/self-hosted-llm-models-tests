# self-hosted-llm-models-terraform

Folder that holds the Terraform files for creating a test cluster on Exoscale SKS using Camptocamp's [DevOps Stack](https://devops-stack.io/).

```bash
# Create the cluster
summon terraform init && summon terraform apply

# Get the kubeconfig settings for the created cluster (https://community.exoscale.com/documentation/sks/quick-start/#kubeconfig)
summon exo compute sks kubeconfig <CLUSTER_NAME> kube-admin --zone de-fra-1 --group system:masters > <PATH_TO_KUBECONFIF_TO_GENERATE.yaml>

# Destroy the cluster
summon terraform state rm $(summon terraform state list | grep "argocd_application\|argocd_project\|argocd_cluster\|argocd_repository\|kubernetes_\|helm_") && summon terraform destroy
```
