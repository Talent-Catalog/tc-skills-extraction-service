resource "aws_cloudwatch_log_group" "logs" {
  name              = "/fargate/service/${var.project_name}-fargate-log"
  retention_in_days = "14"
}
