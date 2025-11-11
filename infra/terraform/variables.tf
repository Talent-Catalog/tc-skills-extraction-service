# variables.tf

variable "project_name" {
  description = "Name of project - all resources will be named based on this"
}

variable "project_description" {
  description = "Description of project"
}

variable "aws_region" {
  description = "The AWS region things are created in"
  default     = "us-east-1"
}

variable "image_tag" {
  description = "The image tag for the ECS service"
  default     = "latest"
}


variable "app_port" {
  description = "Port exposed by the docker image to redirect traffic to"
  default     = 8088
}

variable "health_check_path" {
  default = "/"
}

variable "dns_namespace" {
  description = "Private DNS namespace"
}

variable "fargate_cpu" {
  description = "Fargate instance CPU units to provision (1 vCPU = 1024 CPU units)"
  default     = "1024"
}

variable "fargate_memory" {
  description = "Fargate instance memory to provision (in MiB)"
  default     = "4096"
}

variable "tc_skills_base_url" {
  description = "Talent Catalog skills base url"
}

variable "acm_certificate_arn" {
  description = "The ARN of an ACM certificate"
}
