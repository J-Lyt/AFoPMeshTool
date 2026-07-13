"""Shared logging configuration for the add-on."""

import logging


logger = logging.getLogger("afop_mesh_tool")
if not any(getattr(handler, "_afop_mesh_tool", False) for handler in logger.handlers):
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[AFoPMT] %(levelname)s: %(message)s"))
    handler._afop_mesh_tool = True
    logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.INFO)


def set_debug(enabled):
    """Show or hide debug messages without changing normal informational output."""
    logger.setLevel(logging.DEBUG if enabled else logging.INFO)


__all__ = ("logger", "set_debug")
