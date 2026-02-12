module tc-opc-test {
  source = "./.."
  project_name = "tc-skills-extraction"
  project_description = "OPC staging setup for tc-skills-extraction"
  image_tag = "staging-latest"
  fargate_cpu = 512
  fargate_memory = 2048
  dns_namespace = "tc-skills-extraction.local"
  app_port = 8000
  health_check_path = "/readyz"
  tc_skills_base_url = "https://tctalent-test.org/api/public/skill/names"
  acm_certificate_arn = "<OPC_STAGING_ACM_CERTIFICATE_ARN>"
}

terraform {
  backend "s3" {
    bucket         = "tc-skills-extraction-terraform-state-opc-test"
    key            = "state/terraform.tfstate"
    region         = "eu-west-2"
    encrypt        = true
  }
}
