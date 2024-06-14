output "ingress_domain" {
  description = "The domain to use for accessing the applications."
  value       = "${module.sks.cluster_name}.${module.sks.base_domain}"
}

output "kubernetes_kubeconfig" {
  description = "Configuration that can be copied into `.kube/config in order to access the cluster with `kubectl`."
  value       = module.sks.raw_kubeconfig
  sensitive   = true
}

output "cluster_issuers" {
  description = "Map containing the cluster issuers created by cert-manager."
  value       = module.cert-manager.cluster_issuers
}
