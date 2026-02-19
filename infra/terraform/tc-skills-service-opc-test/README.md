# tc-skills-extraction OPC Staging Environment

Deploy the tc-skills-extraction infrastructure to the OPC AWS staging account (`164804461258`, `eu-west-2`).

## Prerequisites

- AWS CLI configured with credentials that can assume `arn:aws:iam::164804461258:role/opc-staging-terraform-exec`
- Terraform >= 1.0
- The S3 backend bucket (`opc-shared-terraform-state`) and DynamoDB lock table (`opc-terraform-locks`)
  must already exist in the OPC account

## 1. Initialise Terraform

```bash
cd infra/terraform/tc-skills-service-opc-test
terraform init
```

## 2. Set secrets

Create/edit `secrets.auto.tfvars` in this directory with the real password values:

```hcl
a_secret_parameter= "..."
another_secret_parameter= "..."
```

This file is git-ignored and auto-loaded by Terraform. **Do not commit it.**

## 3. Deploy infrastructure

```bash
terraform plan -out tfplan
terraform apply tfplan
```

This creates all infrastructure (VPC, ECS, ALB, ECR, Route53, ACM) and populates SSM parameters
with the values defined in `main.tf`.

The skills extraction service has no secrets — all SSM parameters (`SKILLS_BASE_URL`) are populated
directly from `main.tf` values. No `secrets.auto.tfvars` file is needed.

## SSM parameter updates

To update any of the parameters, simply update the relevant SSM parameter directly in the
AWS console.

Then restart the ECS service to pick up the new values.

ECS tasks that restart (scaling, crashes, deployments) automatically fetch the current SSM values
without needing to re-run Terraform.
