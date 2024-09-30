locals {
  # Secrets can only be created after the first bootstrap of the cluster, because the namespaces only exist after 
  # Argo CD has created them.
  create_k8s_secrets = true

  gheleno_hf_tokens_namespaces = local.create_k8s_secrets ? [
    "gheleno-text-generation-inference",
    "gheleno-chat-ui",
    "gheleno-llama-cpp",
  ] : []
}

resource "kubernetes_secret" "gheleno_hf_token" {
  for_each = toset(local.gheleno_hf_tokens_namespaces)

  metadata {
    name      = "hugging-face-token"
    namespace = each.key
  }

  data = {
    token = var.gheleno_hf_token
  }
}

# resource "kubernetes_secret" "gheleno_chatui_oidc" {
#   count = local.create_k8s_secrets ? 1 : 0

#   metadata {
#     name      = "chatui-oidc"
#     namespace = "gheleno-chat-ui"
#   }

#   data = {
#     OPENID_CONFIG = <<-EOT
#       {
#         PROVIDER_URL: "${module.oidc.oidc.issuer_url}",
#         CLIENT_ID: "${module.oidc.oidc.client_id}",
#         CLIENT_SECRET: "${module.oidc.oidc.client_secret}",
#         NAME_CLAIM: "username",
#         SCOPES: "openid profile email",
#       }
#     EOT
#   }
# }
