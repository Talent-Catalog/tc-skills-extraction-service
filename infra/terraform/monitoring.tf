################################################################################
# ALB CloudWatch Alarms (TC-1526)
################################################################################

resource "aws_cloudwatch_metric_alarm" "alb_unhealthy_hosts" {
  count = var.cloudwatch_alarms_enabled ? 1 : 0

  alarm_name          = "${local.name}-${var.environment}-alb-unhealthy-hosts"
  alarm_description   = "${local.name} (${var.environment}): ALB target group has unhealthy hosts. TC-1526."
  namespace           = "AWS/ApplicationELB"
  metric_name         = "UnHealthyHostCount"
  statistic           = "Maximum"
  comparison_operator = "GreaterThanThreshold"
  threshold           = var.alarm_unhealthy_host_threshold
  period              = 60
  evaluation_periods  = 5
  datapoints_to_alarm = 3
  treat_missing_data  = "notBreaching"

  dimensions = {
    TargetGroup  = module.alb.target_groups["ex_ecs"].arn_suffix
    LoadBalancer = module.alb.arn_suffix
  }

  alarm_actions = [var.alarm_sns_topic_arn]
  ok_actions    = [var.alarm_sns_topic_arn]

  tags = merge(local.tags, { Name = "${local.name}-${var.environment}-alb-unhealthy-hosts" })

  lifecycle {
    precondition {
      condition     = !var.cloudwatch_alarms_enabled || var.alarm_sns_topic_arn != null
      error_message = "alarm_sns_topic_arn must be set when cloudwatch_alarms_enabled is true (TC-1526)."
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "alb_target_response_time_p95" {
  count = var.cloudwatch_alarms_enabled ? 1 : 0

  alarm_name          = "${local.name}-${var.environment}-alb-latency-p95"
  alarm_description   = "${local.name} (${var.environment}): ALB p95 target response time is elevated. TC-1526."
  namespace           = "AWS/ApplicationELB"
  metric_name         = "TargetResponseTime"
  extended_statistic  = "p95"
  comparison_operator = "GreaterThanThreshold"
  threshold           = var.alarm_target_response_time_p95_threshold
  period              = 60
  evaluation_periods  = 5
  datapoints_to_alarm = 5
  treat_missing_data  = "notBreaching"

  dimensions = {
    LoadBalancer = module.alb.arn_suffix
  }

  alarm_actions = [var.alarm_sns_topic_arn]
  ok_actions    = [var.alarm_sns_topic_arn]

  tags = merge(local.tags, { Name = "${local.name}-${var.environment}-alb-latency-p95" })
}

resource "aws_cloudwatch_metric_alarm" "alb_5xx_target" {
  count = var.cloudwatch_alarms_enabled ? 1 : 0

  alarm_name          = "${local.name}-${var.environment}-alb-5xx-target"
  alarm_description   = "${local.name} (${var.environment}): targets behind the ALB are returning HTTP 5XX responses. TC-1526."
  namespace           = "AWS/ApplicationELB"
  metric_name         = "HTTPCode_Target_5XX_Count"
  statistic           = "Sum"
  comparison_operator = "GreaterThanThreshold"
  threshold           = var.alarm_target_5xx_threshold
  period              = 300
  evaluation_periods  = 1
  datapoints_to_alarm = 1
  treat_missing_data  = "notBreaching"

  dimensions = {
    LoadBalancer = module.alb.arn_suffix
  }

  alarm_actions = [var.alarm_sns_topic_arn]
  ok_actions    = [var.alarm_sns_topic_arn]

  tags = merge(local.tags, { Name = "${local.name}-${var.environment}-alb-5xx-target" })
}

resource "aws_cloudwatch_metric_alarm" "alb_5xx_elb" {
  count = var.cloudwatch_alarms_enabled ? 1 : 0

  alarm_name          = "${local.name}-${var.environment}-alb-5xx-elb"
  alarm_description   = "${local.name} (${var.environment}): the ALB itself is generating HTTP 5XX responses. TC-1526."
  namespace           = "AWS/ApplicationELB"
  metric_name         = "HTTPCode_ELB_5XX_Count"
  statistic           = "Sum"
  comparison_operator = "GreaterThanThreshold"
  threshold           = var.alarm_elb_5xx_threshold
  period              = 300
  evaluation_periods  = 1
  datapoints_to_alarm = 1
  treat_missing_data  = "notBreaching"

  dimensions = {
    LoadBalancer = module.alb.arn_suffix
  }

  alarm_actions = [var.alarm_sns_topic_arn]
  ok_actions    = [var.alarm_sns_topic_arn]

  tags = merge(local.tags, { Name = "${local.name}-${var.environment}-alb-5xx-elb" })
}
