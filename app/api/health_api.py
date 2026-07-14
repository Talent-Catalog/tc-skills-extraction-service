from fastapi import APIRouter, Request


router = APIRouter(
  tags=[
    "health",
  ],
)

@router.get("/livez")
def livez(
    request: Request,
) -> dict[str, bool]:
  """
  Report whether application startup completed successfully.

  The ready flag is stored in application state during the FastAPI lifespan.

  This is a commonly used liveness check endpoint.
  :param request: contains the app instance where the state is stored
  :return: a JSON response with the ready status
  """
  ready = bool(getattr(request.app.state, "ready", False))
  return {"ready": ready}
