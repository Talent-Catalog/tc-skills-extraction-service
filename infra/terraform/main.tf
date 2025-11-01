# Use standard Terraform AWS modules where possible.
# See https://registry.terraform.io/browse/modules?provider=aws

# todo Use secrets manager

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.89.0"
    }
  }
}

provider "aws" {
  region = local.region
}

data "aws_availability_zones" "available" {}

locals {
  name    = var.project_name
  description = var.project_description
  region  = var.aws_region

  # This forms the base of our network addresses: the first 16 bits (the 10.0) will be unchanged.
  vpc_cidr = "10.0.0.0/16"

  #This selects three of the AWS existing availability zones
  azs = slice(data.aws_availability_zones.available.names, 0, 3)

  container_name = "${local.name}-container"

  container_port = var.app_port

  tags = {
    Name       = local.name
    Repository = "https://github.com/Talent-Catalog/tc-api"
  }
}

################################################################################
# Cluster
################################################################################

module "ecs_cluster" {
  source  = "terraform-aws-modules/ecs/aws//modules/cluster"
  version = "5.12.0"

  cluster_name = local.name

  # Capacity provider
  fargate_capacity_providers = {
    FARGATE = {
      default_capacity_provider_strategy = {
        weight = 50
        base   = 20
      }
    }
    FARGATE_SPOT = {
      default_capacity_provider_strategy = {
        weight = 50
      }
    }
  }

  tags = local.tags
}

################################################################################
# TC Skills Extraction Service
################################################################################

module "ecs_service" {
  source = "terraform-aws-modules/ecs/aws//modules/service"
  version = "5.12.0"

  name        = local.name
  cluster_arn = module.ecs_cluster.arn

  cpu    = var.fargate_cpu
  memory = var.fargate_memory

  # Enables ECS Exec
  enable_execute_command = true

  # Container definition(s)
  container_definitions = {

    (local.container_name) = {
      cpu       = var.fargate_cpu
      memory    = var.fargate_memory
      essential = true

      image = "${aws_ecr_repository.repo.repository_url}:${var.image_tag}"
      port_mappings = [
        {
          name          = local.container_name
          containerPort = local.container_port
          hostPort      = local.container_port
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "SKILLS_BASE_URL"
          value = var.tc_skills_base_url
        }
      ]

      # Example image used requires access to write to root filesystem
      readonly_root_filesystem = false

      enable_cloudwatch_logging = true
      log_configuration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/fargate/service/${local.name}-fargate-log"
          awslogs-stream-prefix = "ecs"
          awslogs-region        = local.region
        }
      }

      linux_parameters = {
        capabilities = {
          add = []
          drop = [
            "NET_RAW"
          ]
        }
      }

      memory_reservation = 100
    }
  }

  service_connect_configuration = {
    namespace = aws_service_discovery_private_dns_namespace.this.arn
    services = [{
      discovery_name = local.container_name
      port_name      = local.container_name
      client_aliases = []
    }]

  }

  load_balancer = {
    service = {
      target_group_arn = module.alb.target_groups["ex_ecs"].arn
      container_name   = local.container_name
      container_port   = local.container_port
    }
  }

  subnet_ids = module.vpc.private_subnets
  security_group_rules = {
    alb_ingress_3000 = {
      type                     = "ingress"
      from_port                = local.container_port
      to_port                  = local.container_port
      protocol                 = "tcp"
      description              = "Service port"
      source_security_group_id = module.alb.security_group_id
    }
    egress_all = {
      type        = "egress"
      from_port   = 0
      to_port     = 0
      protocol    = "-1"
      cidr_blocks = ["0.0.0.0/0"]
    }
  }

  service_tags = {
    Name = "${local.name}-service"
  }

  tags = local.tags
}

################################################################################
# Supporting Resources
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

resource "aws_ecr_repository" "repo" {
  name                 = local.name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_service_discovery_http_namespace" "this" {
  name        = local.name
  description = "CloudMap namespace for ${local.name}"
  tags        = local.tags
}

resource "aws_service_discovery_private_dns_namespace" "this" {
  name = var.dns_namespace
  description = "Private DNS namespace for ${var.dns_namespace}"
  vpc  = module.vpc.vpc_id
}

module "alb" {
  source  = "terraform-aws-modules/alb/aws"
  version = "9.14.0"

  name = local.name

  load_balancer_type = "application"

  vpc_id  = module.vpc.vpc_id
  subnets = module.vpc.public_subnets

  # For example only
  enable_deletion_protection = false

  # Security Group
  security_group_ingress_rules = {
    all_http = {
      from_port   = 80
      to_port     = 80
      ip_protocol = "tcp"
      cidr_ipv4   = "0.0.0.0/0"
    }
    https = {
      from_port   = 443
      to_port     = 443
      ip_protocol = "tcp"
      cidr_ipv4   = "0.0.0.0/0"
    }
  }
  security_group_egress_rules = {
    all = {
      ip_protocol = "-1"
      cidr_ipv4   = module.vpc.vpc_cidr_block
    }
  }

  listeners = {
    ex_http = {
      port     = 80
      protocol = "HTTP"

      redirect = {
        port        = "443"
        protocol    = "HTTPS"
        status_code = "HTTP_301"
      }
    }
    ex_https = {
      port            = 443
      protocol        = "HTTPS"
      ssl_policy      = "ELBSecurityPolicy-2016-08"
      certificate_arn = var.acm_certificate_arn

      forward = {
        target_group_key = "ex_ecs"
      }
    }
  }

  target_groups = {
    ex_ecs = {
      backend_protocol                  = "HTTP"
      backend_port                      = local.container_port
      target_type                       = "ip"
      deregistration_delay              = 5
      load_balancing_cross_zone_enabled = true

      health_check = {
        enabled             = true
        healthy_threshold   = 2
        interval            = 60
        matcher             = "200"
        path                = var.health_check_path
        port                = "traffic-port"
        protocol            = "HTTP"
        timeout             = 10
        unhealthy_threshold = 5
      }

      # There's nothing to attach here in this definition. Instead,
      # ECS will attach the IPs of the tasks to this target group
      create_attachment = false
    }
  }

  tags = local.tags
}
