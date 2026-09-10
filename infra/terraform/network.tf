################################################################################
# VPC
################################################################################

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.19.0"

  name = local.name
  cidr = local.vpc_cidr

  azs              = local.azs

  # We have three types of subnet: public, private and database.
  # Here is a nice image illustrating those zones: https://miro.medium.com/v2/1*rH2xDaYPE_VOAT8vBKVTug.png
  # We need one of each of those types of subnet in each of the three availability zones
  # cidrsubnet is a standard function which calculates subnets: https://developer.hashicorp.com/terraform/language/functions/cidrsubnet
  public_subnets   = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 8, k)]
  private_subnets  = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 8, k + 3)]

  # Not needed for the Python skills service
  # database_subnets = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 8, k + 6)]
  #
  # create_database_subnet_group = true
  # create_database_subnet_route_table = true
  # create_database_internet_gateway_route = true

  enable_dns_support   = true
  enable_dns_hostnames = true

  enable_nat_gateway = true
  single_nat_gateway = true

  tags = local.tags
}

################################################################################
# Security Group
################################################################################

module "security_group" {
  source  = "terraform-aws-modules/security-group/aws"
  version = "5.3.0"

  name        = local.name
  description = "Talent Catalog M&E security group"
  vpc_id      = module.vpc.vpc_id

  # ingress
  # Not needed by the Python skills service
  # ingress_with_cidr_blocks = [
  #   {
  #     from_port   = 5432
  #     to_port     = 5432
  #     protocol    = "tcp"
  #     description = "PostgreSQL access from within VPC"
  #     cidr_blocks = "0.0.0.0/0"
  #   },
  # ]

  tags = local.tags
}
