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

variable "gheleno_hf_token" {
  description = "Hugging Face API token to use for the Text Generation Inference and ChatUI services."
  type        = string
  sensitive   = true
}
