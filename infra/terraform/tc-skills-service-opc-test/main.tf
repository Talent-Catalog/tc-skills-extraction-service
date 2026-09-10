# Configure the AWS provider
# NOTE: Provider configuration MUST remain here (cannot be moved to parent module).
# Providers cannot have configuration parameters injected via module variables.
# Each environment targets a different AWS account via different assume_role ARNs.
# OPC staging account: 164804461258
provider "aws" {
  region = "eu-west-2"

  assume_role {
    role_arn = "arn:aws:iam::164804461258:role/opc-staging-terraform-exec"
  }
}

# tc-skills-extraction infrastructure for OPC AWS staging account
module "tc-opc-test" {
  source = "./.."

  # Provided as Terraform inputs
  project_name        = "tc-skills-extraction"
  project_description = "OPC staging setup for tc-skills-extraction"
  environment         = "opc-staging"
  aws_region          = "eu-west-2"
  image_tag           = "staging-latest"
  fargate_cpu         = 512
  fargate_memory      = 2048
  dns_namespace       = "tc-skills-extraction.local"
  app_port            = 8000
  health_check_path   = "/readyz"
  site_domain         = "test.skills.tctalent.org"

  # SSM-backed (stored in SSM, injected into ECS task)
  tc_skills_base_url  = "https://tctalent-test.org/api/public/skill/names"
}

# Configure the opc-staging terraform workspace
# NOTE: The terraform block with backend configuration MUST remain in this file (cannot be moved to parent module).
# This is because backend configuration can only exist in the root module where terraform init/apply is run
terraform {
  backend "s3" {
    bucket         = "opc-shared-terraform-state"
    key            = "staging/tc-skills-extraction/terraform.tfstate"
    region         = "eu-west-2"
    dynamodb_table = "opc-terraform-locks"
    encrypt        = true
  }
}
