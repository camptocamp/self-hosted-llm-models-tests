locals {
  llm_apps = {
    app_project_name = "llm-apps"
    repo_url         = "https://github.com/lentidas/self-hosted-llm-models-tests.git"
    apps = [for app in [
      {
        name      = "nvidia-utilities"
        namespace = "nvidia-utilities"
      },
      # {
      #   name      = "text-generation-inference"
      #   namespace = "huggingface-apps"
      # },
      # {
      #   name      = "chat-ui"
      #   namespace = "huggingface-apps"
      # },
      ] : merge(app, {
        repo_url    = "https://github.com/lentidas/self-hosted-llm-models-tests.git"
        repo_branch = "main"
      })
    ]
  }
}

resource "argocd_project" "llm-apps" {
  metadata {
    name      = local.llm_apps.app_project_name
    namespace = "argocd"
  }

  spec {
    description = "Argo CD project for the LLM applications and dependencies (e.g. NVIDIA Device Plugin)"

    source_repos = [
      local.llm_apps.repo_url
    ]

    dynamic "destination" {
      for_each = concat(
        [{ namespace = "argocd" }], # The destination for the ApplicationSet that creates all the applications.
        [for app in local.llm_apps.apps : { namespace = app.namespace }]
      )
      content {
        name      = "in-cluster"
        namespace = destination.value["namespace"]
      }
    }

    orphaned_resources {
      warn = true
    }

    cluster_resource_whitelist {
      group = "*"
      kind  = "*"
    }
  }
}

resource "argocd_application_set" "llm-apps" {
  metadata {
    name      = local.llm_apps.app_project_name
    namespace = "argocd"
  }

  spec {
    generator {
      list {
        elements = local.llm_apps.apps
      }
    }

    template {
      metadata {
        name      = "{{name}}"
        namespace = "argocd"
        labels = {
          "application-set" = local.llm_apps.app_project_name
          "application"     = "{{name}}"
          "cluster"         = "in-cluster"
          "namespace"       = "{{namespace}}"
        }
      }

      spec {
        # Refer to the Argo CD project created above and not the local with the same name in order to create an 
        # implicit dependency.
        project = resource.argocd_project.llm-apps.metadata.0.name

        source {
          repo_url        = "{{repo_url}}"
          target_revision = "{{repo_branch}}"
          path            = "charts/apps/{{name}}"
        }

        destination {
          name      = "in-cluster"
          namespace = "{{namespace}}"
        }

        sync_policy {
          automated {
            prune       = true
            self_heal   = true
            allow_empty = true
          }

          retry {
            backoff {
              duration     = "20s"
              max_duration = "2m"
              factor       = "2"
            }
            limit = "5"
          }

          sync_options = [
            "CreateNamespace=true"
          ]
        }
      }
    }
  }
}
