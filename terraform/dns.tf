# Requires a subscription to Exoscale DNS service, which should be mannually activated on the web console.
# If using nip.io, which is deployed automatically, both these resources are not needed.

# Since the is-sandbox-exo.camptocamp.com domain was added manually to the sandbox account for everyone, we use a `data`
# instead of `resource` to avoid conflicts.
data "exoscale_domain" "domain" {
  name = local.base_domain
}

# This resource should be deactivated if there are multiple development clusters on the same account.
resource "exoscale_domain_record" "wildcard" {
  count = local.activate_wildcard_record ? 1 : 0

  domain      = data.exoscale_domain.domain.id
  name        = local.subdomain != "" ? "*.${local.subdomain}" : "*"
  record_type = "A"
  ttl         = "300"
  content     = module.sks.nlb_ip_address
}
