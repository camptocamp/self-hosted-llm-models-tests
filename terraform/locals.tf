locals {
  kubernetes_version       = "1.30.2"
  cluster_name             = "llm-models-sks" # Must be unique for each DevOps Stack deployment in a single account.
  zone                     = "de-fra-1"
  service_level            = "starter"
  base_domain              = "is-sandbox-exo.camptocamp.com"
  subdomain                = ""
  activate_wildcard_record = false
  cluster_issuer           = module.cert-manager.cluster_issuers.production
  letsencrypt_issuer_email = "letsencrypt@camptocamp.com"
  enable_service_monitor   = false # Can be enabled after the first bootstrap.
  app_autosync             = true ? { allow_empty = false, prune = true, self_heal = true } : {}
}
