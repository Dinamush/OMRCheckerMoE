"""OMRChecker Web UI and JSON API.

A thin FastAPI wrapper around the OMRChecker engine. The package exposes:

- A JSON API under ``/api/v1/*`` suitable for headless/automated use.
- A minimal Jinja2-based UI for manually managing batches.

The UI and API share a single service layer (``webui.services``) so any
functionality available via HTML is also available via the API.
"""

from webui.app import app, create_app

__all__ = ["app", "create_app"]
