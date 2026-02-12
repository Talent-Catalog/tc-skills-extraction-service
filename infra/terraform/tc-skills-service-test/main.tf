provider "aws" {
  region = "us-east-1"
}

module tc-test {
  source = "./.."
  project_name = "tc-skills-extraction"
  project_description = "Staging setup for tc-skills-extraction"
  environment = "staging"
  image_tag = "staging-latest"
  fargate_cpu = 512
  fargate_memory = 2048
  dns_namespace = "tc-skills-extraction.local"
  app_port = 8000
  health_check_path = "/readyz"
  site_domain = "tctalent-test.org"

  # SSM parameter values — initially set here, but can be subsequently updated directly
  # in AWS SSM Parameter Store without requiring a Terraform apply
  tc_skills_base_url = "https://tctalent-test.org/api/public/skill/names"
}

terraform {
  backend "s3" {
    bucket         = "tc-skills-extraction-terraform-state"
    key            = "state/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
  }
}
