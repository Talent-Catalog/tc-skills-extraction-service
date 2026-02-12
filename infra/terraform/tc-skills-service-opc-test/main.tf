module tc-opc-test {
  source = "./.."
  project_name = "tc-skills-extraction"
  project_description = "OPC staging setup for tc-skills-extraction"
  aws_region = "eu-west-2"
  image_tag = "staging-latest"
  fargate_cpu = 512
  fargate_memory = 2048
  dns_namespace = "tc-skills-extraction.local"
  app_port = 8000
  health_check_path = "/readyz"
  tc_skills_base_url = "https://tctalent-test.org/api/public/skill/names"
  acm_certificate_arn = "<OPC_STAGING_ACM_CERTIFICATE_ARN>"
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
