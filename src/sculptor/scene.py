"""Helpers for the Swift scene."""

from pathlib import Path
from typing import Any

import spatialgeometry as sg

# Kept before fix_mesh_paths replaces it, so calling the fix twice does no harm
_mesh_to_dict = sg.Mesh.to_dict


def _mesh_to_dict_posix(self: sg.Mesh) -> dict[str, Any]:
    shape = _mesh_to_dict(self)
    if shape.get("filename"):
        shape["filename"] = Path(shape["filename"]).as_posix()
    return shape


def fix_mesh_paths() -> None:
    """Send the paths of the meshes to Swift with forward slashes.

    On Windows the paths have backslashes, and the Swift page asks its file server for them
    as they are. The server only understands forward slashes, so no mesh loads, not even the
    robot. Call this once before adding anything to Swift. On macOS and Linux it changes nothing.
    """
    sg.Mesh.to_dict = _mesh_to_dict_posix
