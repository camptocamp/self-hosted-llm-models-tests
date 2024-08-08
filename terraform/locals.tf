locals {
  kubernetes_version       = "1.30.3"
  cluster_name             = "llm-models-sks" # Must be unique for each DevOps Stack deployment in a single account.
  zone                     = "de-fra-1"
  service_level            = "starter"
  base_domain              = "is-ai.camptocamp.com"
  subdomain                = ""
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
        auto_sync = true
        list = [
          {
            name            = "text-generation-inference"
            target_revision = "main"
            chart_repo_url  = "https://github.com/camptocamp/self-hosted-llm-models-charts.git"
            values_repo_url = "https://github.com/lentidas/self-hosted-llm-models-values.git"
          },
          # {
          #   name            = "chat-ui"
          #   target_revision = "main"
          #   chart_repo_url  = "https://github.com/camptocamp/self-hosted-llm-models-charts.git"
          #   values_repo_url = "https://github.com/lentidas/self-hosted-llm-models-values.git"
          # }
        ]
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
        # auto_sync = true
        # list = [
        #   {
        #     name            = "llama-cpp"
        #     target_revision = "main"
        #     chart_repo_url  = "https://github.com/camptocamp/self-hosted-llm-models-charts.git"
        #     values_repo_url = "https://github.com/chornberger-c2c/self-hosted-llm-models-values.git"
        #   },
        # ]
      }
    }
  }
}
