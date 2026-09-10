################################################################################
# Route53 DNS
################################################################################

resource "aws_route53_zone" "this" {
  name = var.site_domain

  tags = local.tags
}

# A record pointing the domain to the ALB
resource "aws_route53_record" "ipv4" {
  zone_id = aws_route53_zone.this.zone_id
  name    = var.site_domain
  type    = "A"

  alias {
    name                   = module.alb.dns_name
    zone_id                = module.alb.zone_id
    evaluate_target_health = false
  }
}

resource "aws_route53_record" "www" {
  zone_id = aws_route53_zone.this.zone_id
  name    = "www"
  type    = "CNAME"
  ttl     = "600"
  records = [var.site_domain]
}
