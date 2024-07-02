locals {
  llm_apps = {
    source_repos = [
      "https://github.com/camptocamp/self-hosted-llm-models-tests.git", # this repo
      "https://github.com/camptocamp/self-hosted-llm-models-charts.git" # the repo where the generalized charts are stored
    ]
  }
}

resource "null_resource" "llm-apps" {
  triggers = {
    argocd = module.argocd.id
  }
}

resource "argocd_project" "llm-apps" {
  depends_on = [null_resource.llm-apps]

  metadata {
    name      = "llm-apps"
    namespace = "argocd"
  }

  spec {
    description = "LLM applications and their dependencies (e.g. NVIDIA Device Plugin)"

    source_repos = local.llm_apps.source_repos

    destination {
      name      = "in-cluster"
      namespace = "*"
    }

    destination {
      server    = "https://kubernetes.default.svc"
      namespace = "*"
    }

    orphaned_resources {
      warn = true

      # Because we are creating the Argo CD resources using Terraform, these appear as orphaned resources in the Argo CD
      # interface, so we need to ignore them.
      ignore {
        group = "argoproj.io"
        kind  = "*"
        name  = "*"
      }

      # Most of the Secrets are created by the External Secrets operator, so it's best they are ignored.
      ignore {
        kind = "Secret"
        name = "*"
      }
    }
    cluster_resource_whitelist {
      group = "*"
      kind  = "*"
    }

    namespace_resource_whitelist {
      group = "*"
      kind  = "*"
    }
  }
}

resource "argocd_application" "llm-apps" {
  depends_on = [null_resource.llm-apps]

  metadata {
    name      = "llm-apps"
    namespace = "argocd"
  }

  spec {
    project = resource.argocd_project.llm-apps.metadata.0.name

    source {
      repo_url        = local.llm_apps.source_repos[0]
      target_revision = "main"
      path            = "llm-apps"
      helm {
        value_files = [
          "values.yaml"
        ]
        values = yamlencode({
          workEnvironments = { for k, v in local.engineers :
            k => {
              name = format("%s %s", v.oidc.first_name, v.oidc.last_name)
              apps = v.apps
            }
          }
        })
      }
    }

    destination {
      name      = "in-cluster"
      namespace = "argocd"
    }

    sync_policy {
      automated {
        prune       = true
        self_heal   = true
        allow_empty = true
      }
    }
  }
}
