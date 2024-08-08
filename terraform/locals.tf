locals {
  kubernetes_version       = "1.30.3"
  cluster_name             = "llm-models-sks" # Must be unique for each DevOps Stack deployment in a single account.
  zone                     = "de-fra-1"
  service_level            = "starter"
  base_domain              = "is-ai.camptocamp.com"
  subdomain                = ""
  domain                   = format("llama-cpp.%s", trimprefix("${local.subdomain}.${local.base_domain}", "."))
  domain_full              = format("llama-cpp.%s.%s", trimprefix("${local.subdomain}.${local.cluster_name}", "."), local.base_domain)
  activate_wildcard_record = true
  cluster_issuer           = module.cert-manager.cluster_issuers.production
  letsencrypt_issuer_email = "letsencrypt@camptocamp.com"
  enable_service_monitor   = false # Can be enabled after the first bootstrap.
  app_autosync             = true ? { allow_empty = false, prune = true, self_heal = true } : {}

  engineers = {
    gheleno = {
      oidc = {
        username   = "gheleno"
        email      = "goncalo.heleno@camptocamp.com"
        first_name = "Gonçalo"
        last_name  = "Heleno"
      }
      apps = {
        #  auto_sync = true
        #  list = [
        #    {
        #      name            = "text-generation-inference"
        #      target_revision = "main"
        #      chart_repo_url  = "https://github.com/camptocamp/self-hosted-llm-models-charts.git"
        #      values_repo_url = "https://github.com/lentidas/self-hosted-llm-models-values.git"
        #    },
        #    # {
        #    #   name            = "chat-ui"
        #    #   target_revision = "main"
        #    #   chart_repo_url  = "https://github.com/camptocamp/self-hosted-llm-models-charts.git"
        #    #   values_repo_url = "https://github.com/lentidas/self-hosted-llm-models-values.git"
        #    # }
        #  ]
      }
    }
    bquartier = {
      oidc = {
        username   = "bquartier"
        email      = "benoit.quartier@camptocamp.com"
        first_name = "Benoît"
        last_name  = "Quartier"
      }
      apps = {}
    }
    chornberger = {
      oidc = {
        username   = "chornberger"
        email      = "christopher.hornberger@camptocamp.com"
        first_name = "Christopher"
        last_name  = "Hornberger"
      }
      apps = {
        auto_sync = true
        list = [
          {
            name            = "llama-cpp"
            target_revision = "main"
            chart_repo_url  = "https://github.com/camptocamp/self-hosted-llm-models-charts.git"
            values_repo_url = "https://github.com/chornberger-c2c/self-hosted-llm-models-values.git"
          },
        ]
      }
    }
  }
  helm_values = [{
    oidc = var.oidc != null ? {
      oauth2_proxy_image      = "quay.io/oauth2-proxy/oauth2-proxy:v7.6.0"
      issuer_url              = var.oidc.issuer_url
      redirect_url            = format("https://%s/oauth2/callback", local.domain_full)
      client_id               = var.oidc.client_id
      client_secret           = var.oidc.client_secret
      cookie_secret           = resource.random_string.oauth2_cookie_secret.result
      oauth2_proxy_extra_args = var.oidc.oauth2_proxy_extra_args
    } : null
    ingress = {
      enabled = var.enable_dashboard_ingress
      hosts = [
        local.domain,
        local.domain_full
      ]
    }
  }]
}
resource "random_string" "oauth2_cookie_secret" {
  length  = 32
  special = false
}
