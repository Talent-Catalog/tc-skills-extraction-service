from fastapi import APIRouter, Request


router = APIRouter(
  tags=[
    "health",
  ],
)

@router.get("/livez")
@router.get("/readyz")
def livez(
    request: Request,
) -> dict[str, bool]:
  """
  Report whether application startup completed successfully.

  The ready flag is stored in application state during the FastAPI lifespan.

  This is a commonly used liveness check endpoint.

  Also registered as /readyz: every environment's ALB target group health
  check is still configured with that path (see infra/terraform/*/main.tf),
  and this route was renamed from /readyz to /livez in commit 20e1468
  without updating them, which silently broke health checks for every
  deployment since - the ELB marked every new task unhealthy and ECS kept
  rolling back to the last task built before that rename.

  :param request: contains the app instance where the state is stored
  :return: a JSON response with the ready status
  """
  ready = bool(getattr(request.app.state, "ready", False))
  return {"ready": ready}
