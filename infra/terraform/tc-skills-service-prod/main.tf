provider "aws" {
  region = "us-east-1"
}

# SNS topic for CloudWatch alarm notifications (TC-1526).
# Created manually in this TBB prod account console per ADR 002 (talentcatalog repo);
# must already exist before this root is planned.
data "aws_sns_topic" "alarms" {
  name = "tbb-prod-alarms"
}

# tc-skills-extraction infrastructure for TBB AWS production account
module tc-test {
  source = "./.."

  # Provided as Terraform inputs
  project_name        = "tc-skills-extraction"
  project_description = "Production setup for tc-skills-extraction"
  environment         = "production"
  image_tag           = "production-latest"
  fargate_cpu         = 512
  fargate_memory      = 2048
  dns_namespace       = "tc-skills-extraction.local"
  app_port            = 8000
  health_check_path   = "/readyz"
  site_domain         = "tctalent.org"

  # SSM-backed (stored in SSM, injected into ECS task)
  tc_skills_base_url  = "https://tctalent.org/api/public/skill/names"

  # CloudWatch alarm coverage (TC-1526)
  cloudwatch_alarms_enabled = true
  alarm_sns_topic_arn       = data.aws_sns_topic.alarms.arn
}

terraform {
  backend "s3" {
    bucket         = "tc-skills-extraction-terraform-state-prod"
    key            = "state/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
  }
}
