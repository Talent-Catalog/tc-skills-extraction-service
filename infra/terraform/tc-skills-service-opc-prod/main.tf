# Configure the AWS provider
# NOTE: Provider configuration MUST remain here (cannot be moved to parent module).
# Providers cannot have configuration parameters injected via module variables.
# Each environment targets a different AWS account via different assume_role ARNs.
# OPC production account: 289896345557
provider "aws" {
  region = "eu-west-2"

  assume_role {
    role_arn = "arn:aws:iam::289896345557:role/opc-prod-terraform-exec"
  }
}

# tc-skills-extraction infrastructure for OPC AWS production account
module "tc-opc-prod" {
  source = "./.."

  # Provided as Terraform inputs
  project_name        = "tc-skills-extraction"
  project_description = "OPC production setup for tc-skills-extraction"
  environment         = "opc-production"
  aws_region          = "eu-west-2"
  image_tag           = "production-latest"
  fargate_cpu         = 512
  fargate_memory      = 2048
  dns_namespace       = "tc-skills-extraction.local"
  app_port            = 8000
  health_check_path   = "/readyz"
  site_domain         = "skills.plus.tctalent.org"

  # SSM-backed (stored in SSM, injected into ECS task)
  tc_skills_base_url  = "https://plus.tctalent.org/api/public/skill/names"
}

# Configure the opc-production terraform workspace
# NOTE: The terraform block with backend configuration MUST remain in this file (cannot be moved to parent module).
# This is because backend configuration can only exist in the root module where terraform init/apply is run
terraform {
  backend "s3" {
    bucket         = "opc-shared-terraform-state"
    key            = "prod/tc-skills-extraction/terraform.tfstate"
    region         = "eu-west-2"
    dynamodb_table = "opc-terraform-locks"
    encrypt        = true
  }
}
