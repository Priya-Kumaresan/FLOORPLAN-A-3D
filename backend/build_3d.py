import os
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix


# simple world scale: 512px == 10m  → 0.0195 m/px
PX_TO_M = 10.0 / 512.0
WALL_HEIGHT = 3.0       # meters
WALL_THICKNESS = 0.2    # meters


def extrude_walls(lines):
    """
    lines: [((x1,y1),(x2,y2)), ...] in image pixel coords (512×512)
    returns: trimesh.Trimesh (or Scene)
    """
    meshes = []

    for (p1, p2) in lines:
        x1, y1 = p1
        x2, y2 = p2

        # convert pixel coords to meters
        x1m, y1m = x1 * PX_TO_M, y1 * PX_TO_M
        x2m, y2m = x2 * PX_TO_M, y2 * PX_TO_M

        dx = x2m - x1m
        dy = y2m - y1m
        length = np.linalg.norm([dx, dy])
        if length < 0.1:  # ignore tiny
            continue

        angle = np.arctan2(dy, dx)

        # create a box (length × thickness × height)
        wall = trimesh.creation.box(extents=[length, WALL_THICKNESS, WALL_HEIGHT])

        # move pivot to one end of the wall
        wall.apply_translation([-length / 2.0, 0.0, WALL_HEIGHT / 2.0])

        # rotate around Z
        R = rotation_matrix(angle, [0, 0, 1])
        wall.apply_transform(R)

        # translate to correct world position
        wall.apply_translation([x1m, y1m, 0.0])

        meshes.append(wall)

    if not meshes:
        return trimesh.Trimesh()

    merged = trimesh.util.concatenate(meshes)
    return merged


def export_to_glb(mesh, out_name: str = "scene.glb"):
    out_path = os.path.join(os.path.dirname(__file__), out_name)
    mesh.export(out_path)
    return out_path
