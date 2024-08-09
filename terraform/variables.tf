variable "exoscale_iam_key" {
  description = "Exoscale IAM access key to use for the S3 provider."
  type        = string
  sensitive   = true
}

variable "exoscale_iam_secret" {
  description = "Exoscale IAM access secret to use for the S3 provider."
  type        = string
  sensitive   = true
}

variable "prometheus_helm_values_override" {
  description = "Override values for prometheus helm chart."
  type        = any
  default     = {}
}


variable "oidc" {
  description = "OIDC settings to configure OAuth2-Proxy which will be used to protect llama.cpp's dashboard."
  type = object({
    issuer_url              = string
    oauth_url               = optional(string, "")
    token_url               = optional(string, "")
    api_url                 = optional(string, "")
    client_id               = string
    client_secret           = string
    oauth2_proxy_extra_args = optional(list(string), [])
  })
  default = null
}

variable "enable_dashboard_ingress" {
  description = "Boolean to enable the creation of an ingress for the llama.cpp's dashboard. **If enabled, you must provide a value for `base_domain`.**"
  type        = bool
  default     = true
}