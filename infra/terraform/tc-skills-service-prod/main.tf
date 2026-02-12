provider "aws" {
  region = "us-east-1"
}

module tc-test {
  source = "./.."
  project_name = "tc-skills-extraction"
  project_description = "Production setup for tc-skills-extraction"
  environment = "production"
  image_tag = "production-latest"
  fargate_cpu = 512
  fargate_memory = 2048
  dns_namespace = "tc-skills-extraction.local"
  app_port = 8000
  health_check_path = "/readyz"
  acm_certificate_arn = "arn:aws:acm:us-east-1:968457613372:certificate/5dd8d298-5460-4396-ad5e-24e3a2dfa774"

  # SSM parameter values — initially set here, but can be subsequently updated directly
  # in AWS SSM Parameter Store without requiring a Terraform apply
  tc_skills_base_url = "https://tctalent.org/api/public/skill/names"
}

terraform {
  backend "s3" {
    bucket         = "tc-skills-extraction-terraform-state-prod"
    key            = "state/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
  }
}
