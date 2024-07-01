locals {
  llm_apps = {
    repo_url = "https://github.com/lentidas/self-hosted-llm-models-tests.git"
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
    description = "Argo CD project for the LLM applications and dependencies (e.g. NVIDIA Device Plugin)"

    source_repos = [
      local.llm_apps.repo_url
    ]

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
      repo_url        = local.llm_apps.repo_url
      target_revision = "main"
      path            = "llm-apps"
      helm {
        value_files = [
          "values.yaml"
        ]
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
