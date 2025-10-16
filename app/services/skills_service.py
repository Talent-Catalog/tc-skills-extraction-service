import argparse, json, sys
from typing import Dict, Iterable, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from typing import List

class SkillsService:
  def __init__(self):
    print("hello")

  def get_skills(self) -> List[str]:
    #Example: this should be populated from skills on our Postgres database
    # populated from ESCO.
    # Could be around 20,000 of these
    return ["Java", "Spring Boot", "Python", "FastAPI", "Docker", "C++", "Kubernetes", "PostgreSQL",
            "MapStruct", "Angular", "AWS", "Terraform", "Natural language processing","Spring"]

  def make_session(self, token: Optional[str], timeout: float) -> requests.Session:
    s = requests.Session()
    if token:
      s.headers.update({"Authorization": f"Bearer {token}"})
    retry = Retry(
      total=5,
      backoff_factor=0.3,
      status_forcelist=(429, 500, 502, 503, 504),
      allowed_methods=frozenset(["GET"])
    )
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.request = self._with_timeout(s.request, timeout)
    return s

  def _with_timeout(self, request_func, timeout: float):
    def wrapped(method, url, **kwargs):
      kwargs.setdefault("timeout", timeout)
      return request_func(method, url, **kwargs)
    return wrapped

  def iter_all_items_pagejson(self,
      base_url: str,
      size: int = 200,
      params: Optional[Dict] = None,
      token: Optional[str] = None,
      max_pages: int = 10000,
      timeout: float = 15.0
  ) -> Iterable[dict]:
    """
    Iterate all items from a Spring Page<T> endpoint (no HAL).
    Expects JSON fields like: content, number, size, totalPages, last (optional).
    """
    session = self.make_session(token, timeout)
    qp = dict(params or {})
    qp.setdefault("page", 0)      # Spring is 0-based
    qp["size"] = size

    pages = 0
    while pages < max_pages:
      r = session.get(base_url, params=qp)
      r.raise_for_status()
      data = r.json()

      content = data.get("content", [])
      for row in content:
        yield row

      pages += 1

      # Stop conditions
      last = data.get("last")
      if isinstance(last, bool) and last:
        break

      number = data.get("number")
      total_pages = data.get("totalPages")
      if number is not None and total_pages is not None:
        if number + 1 >= total_pages:
          break
        qp["page"] = number + 1
      else:
        # If server doesn’t give enough metadata, bail out to avoid infinite loop.
        break
