"""Service layer shared by the JSON API and the HTML views.

All filesystem and engine interactions live here so that HTTP handlers
stay thin and so the same operations are available to any client.
"""

from webui.services import batches, omr

__all__ = ["batches", "omr"]
