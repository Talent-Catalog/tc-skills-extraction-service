module tc-test {
  source = "./.."
  project_name = "tc-skills-extraction"
  project_description = "Staging setup for tc-skills-extraction"
  # image_tag = "1.0.1-SNAPSHOT"
  image_tag = "staging-latest"
  fargate_cpu = 512
  fargate_memory = 2048
  dns_namespace = "tc-skills-extraction.local"
  app_port = 8000
  health_check_path = "/readyz"
  tc_skills_base_url = "https://tctalent-test.org/api/public/skill/names"
  acm_certificate_arn = "arn:aws:acm:us-east-1:231168606641:certificate/3a502945-f505-46f9-aa08-523c2be2593d"
}

terraform {
  backend "s3" {
    bucket         = "tc-skills-extraction-terraform-state"
    key            = "state/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
  }
}
