################################################################################
# SSM Parameters
################################################################################

resource "aws_ssm_parameter" "skills_base_url" {
  name  = "/${var.project_name}/${var.environment}/SKILLS_BASE_URL"
  type  = "String"
  value = var.tc_skills_base_url
}
