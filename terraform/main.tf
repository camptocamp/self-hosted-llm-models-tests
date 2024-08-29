module "sks" {
  source = "git::https://github.com/camptocamp/devops-stack-module-cluster-sks.git?ref=v1.2.1"

  cluster_name       = local.cluster_name
  kubernetes_version = local.kubernetes_version
  zone               = local.zone
  base_domain        = data.exoscale_domain.domain.name
  subdomain          = local.subdomain

  cni = "calico"

  service_level = local.service_level

  nodepools = {
    "${local.cluster_name}-default" = {
      size            = 3
      instance_type   = "standard.large"
      description     = "Default node pool for ${local.cluster_name}."
      instance_prefix = "default"
      disk_size       = 400
    },
    "${local.cluster_name}-llm" = {
      size            = 1
      instance_type   = "gpu3.small"
      description     = "GPU3 node pool for ${local.cluster_name} for LLM deployment."
      instance_prefix = "llm"
      disk_size       = 400
      labels = {
        "role" = "llm"
      }
      taints = {
        nodepool = "llm:NoSchedule"
      }
    },
  }
}

module "oidc" {
  source = "git::https://github.com/camptocamp/devops-stack-module-oidc-aws-cognito.git?ref=v1.1.1"

  cluster_name = module.sks.cluster_name
  base_domain  = module.sks.base_domain
  subdomain    = local.subdomain

  create_pool = true

  user_map = { for k, v in local.engineers : k => v.oidc }

  callback_urls = [
    format("https://longhorn.%s/oauth2/callback", trimprefix("${local.subdomain}.${local.base_domain}", ".")),
    format("https://longhorn.%s.%s/oauth2/callback", trimprefix("${local.subdomain}.${local.cluster_name}", "."), local.base_domain),
  ]
}

module "argocd_bootstrap" {
  source = "git::https://github.com/camptocamp/devops-stack-module-argocd.git//bootstrap?ref=v6.3.0"

  argocd_projects = {
    "${module.sks.cluster_name}" = {
      destination_cluster = "in-cluster"
    }
  }

  depends_on = [module.sks]
}

# module "secrets" {
#   # source = "git::https://github.com/lentidas/devops-stack-module-secrets.git//aws_secrets_manager?ref=feat/initial_implementation"
#   source = "../../devops-stack-module-secrets/aws_secrets_manager"
#   # source = "../../devops-stack-module-secrets/k8s_secrets"

#   target_revision = "feat/initial_implementation"

#   cluster_name   = module.sks.cluster_name
#   base_domain    = module.sks.base_domain
#   argocd_project = module.sks.cluster_name

#   app_autosync           = local.app_autosync
#   enable_service_monitor = local.enable_service_monitor

#   aws_iam_access_key = {
#     create_iam_access_key = true
#   }

#   dependency_ids = {
#     argocd = module.argocd_bootstrap.id
#   }
# }

module "traefik" {
  source = "git::https://github.com/camptocamp/devops-stack-module-traefik.git//sks?ref=v8.1.0"

  argocd_project = module.sks.cluster_name

  nlb_id                  = module.sks.nlb_id
  router_nodepool_id      = module.sks.router_nodepool_id
  router_instance_pool_id = module.sks.router_instance_pool_id

  app_autosync           = local.app_autosync
  enable_service_monitor = local.enable_service_monitor

  dependency_ids = {
    argocd = module.argocd_bootstrap.id
  }
}

module "cert-manager" {
  source = "git::https://github.com/camptocamp/devops-stack-module-cert-manager.git//sks?ref=v8.6.0"

  argocd_project = module.sks.cluster_name

  letsencrypt_issuer_email = local.letsencrypt_issuer_email

  app_autosync           = local.app_autosync
  enable_service_monitor = local.enable_service_monitor

  dependency_ids = {
    argocd = module.argocd_bootstrap.id
  }
}

module "longhorn" {
  source = "git::https://github.com/camptocamp/devops-stack-module-longhorn.git?ref=v3.6.0"

  cluster_name   = module.sks.cluster_name
  base_domain    = module.sks.base_domain
  subdomain      = local.subdomain
  cluster_issuer = local.cluster_issuer
  argocd_project = module.sks.cluster_name

  app_autosync           = local.app_autosync
  enable_service_monitor = local.enable_service_monitor

  enable_preupgrade_check  = false
  enable_dashboard_ingress = true
  oidc                     = module.oidc.oidc

  tolerations = [
    {
      key      = "nodepool"
      operator = "Equal"
      value    = "llm"
      effect   = "NoSchedule"
    },
  ]

  enable_pv_backups = true
  backup_storage = {
    bucket_name = resource.aws_s3_bucket.this["longhorn"].id
    region      = resource.aws_s3_bucket.this["longhorn"].region
    endpoint    = "sos-${resource.aws_s3_bucket.this["longhorn"].region}.exo.io"
    access_key  = resource.exoscale_iam_api_key.s3_iam_api_key["longhorn"].key
    secret_key  = resource.exoscale_iam_api_key.s3_iam_api_key["longhorn"].secret
  }

  dependency_ids = {
    argocd       = module.argocd_bootstrap.id
    traefik      = module.traefik.id
    cert-manager = module.cert-manager.id
    oidc         = module.oidc.id
  }
}

module "loki-stack" {
  source = "git::https://github.com/camptocamp/devops-stack-module-loki-stack.git//sks?ref=v9.0.0"

  argocd_project = module.sks.cluster_name

  app_autosync = local.app_autosync

  logs_storage = {
    bucket_name = resource.aws_s3_bucket.this["loki"].id
    region      = resource.aws_s3_bucket.this["loki"].region
    access_key  = resource.exoscale_iam_api_key.s3_iam_api_key["loki"].key
    secret_key  = resource.exoscale_iam_api_key.s3_iam_api_key["loki"].secret
  }

  dependency_ids = {
    argocd   = module.argocd_bootstrap.id
    longhorn = module.longhorn.id
  }
}

module "thanos" {
  source = "git::https://github.com/camptocamp/devops-stack-module-thanos.git//sks?ref=v6.0.0"

  cluster_name   = module.sks.cluster_name
  base_domain    = module.sks.base_domain
  subdomain      = local.subdomain
  cluster_issuer = local.cluster_issuer
  argocd_project = module.sks.cluster_name

  app_autosync           = local.app_autosync
  enable_service_monitor = local.enable_service_monitor

  metrics_storage = {
    bucket_name = resource.aws_s3_bucket.this["thanos"].id
    region      = resource.aws_s3_bucket.this["thanos"].region
    access_key  = resource.exoscale_iam_api_key.s3_iam_api_key["thanos"].key
    secret_key  = resource.exoscale_iam_api_key.s3_iam_api_key["thanos"].secret
  }

  thanos = {
    oidc = module.oidc.oidc
  }

  dependency_ids = {
    argocd       = module.argocd_bootstrap.id
    traefik      = module.traefik.id
    cert-manager = module.cert-manager.id
    oidc         = module.oidc.id
    longhorn     = module.longhorn.id
  }
}

module "kube-prometheus-stack" {
  source = "git::https://github.com/camptocamp/devops-stack-module-kube-prometheus-stack.git//sks?ref=v11.1.1"

  cluster_name   = module.sks.cluster_name
  base_domain    = module.sks.base_domain
  subdomain      = local.subdomain
  cluster_issuer = local.cluster_issuer
  argocd_project = module.sks.cluster_name
  # secrets_names  = module.secrets.secrets_names

  app_autosync = local.app_autosync

  metrics_storage = {
    bucket_name = resource.aws_s3_bucket.this["thanos"].id
    region      = resource.aws_s3_bucket.this["thanos"].region
    access_key  = resource.exoscale_iam_api_key.s3_iam_api_key["thanos"].key
    secret_key  = resource.exoscale_iam_api_key.s3_iam_api_key["thanos"].secret
  }

  prometheus = {
    oidc = module.oidc.oidc
  }
  alertmanager = {
    oidc = module.oidc.oidc
  }
  grafana = {
    oidc = module.oidc.oidc
  }

  helm_values = [
    yamldecode(file("${path.module}/helm_values/kube-prometheus-stack.yaml")),
    var.prometheus_helm_values_override
  ]

  dependency_ids = {
    argocd = module.argocd_bootstrap.id
    # secrets      = module.secrets.id
    traefik      = module.traefik.id
    cert-manager = module.cert-manager.id
    oidc         = module.oidc.id
    longhorn     = module.longhorn.id
    loki-stack   = module.loki-stack.id
  }
}

module "argocd" {
  source = "git::https://github.com/camptocamp/devops-stack-module-argocd.git?ref=v6.1.0"

  cluster_name   = module.sks.cluster_name
  base_domain    = module.sks.base_domain
  subdomain      = local.subdomain
  cluster_issuer = local.cluster_issuer
  argocd_project = module.sks.cluster_name

  accounts_pipeline_tokens = module.argocd_bootstrap.argocd_accounts_pipeline_tokens
  server_secretkey         = module.argocd_bootstrap.argocd_server_secretkey

  app_autosync = local.app_autosync

  admin_enabled = false
  exec_enabled  = true

  oidc = {
    name         = "Cognito"
    issuer       = module.oidc.oidc.issuer_url
    clientID     = module.oidc.oidc.client_id
    clientSecret = module.oidc.oidc.client_secret
    requestedIDTokenClaims = {
      groups = {
        essential = true
      }
    }
    requestedScopes = [
      "openid", "profile", "email"
    ]
  }

  rbac = {
    policy_csv = <<-EOT
      g, pipeline, role:admin
      g, devops-stack-admins, role:admin
    EOT
  }

  dependency_ids = {
    argocd                = module.argocd_bootstrap.id
    traefik               = module.traefik.id
    cert-manager          = module.cert-manager.id
    oidc                  = module.oidc.id
    kube-prometheus-stack = module.kube-prometheus-stack.id
  }
}
