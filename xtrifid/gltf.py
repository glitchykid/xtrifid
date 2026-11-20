"""Self-contained glTF 2.0 assets with indexed Game Ready meshes and embedded maps."""

import hashlib
import json
import os
import shutil
import struct
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import __version__
from .geometry import build, check_scene_budget, rotation_matrix
from .model import MAX_PARTS, Scene
from .storage import _sync_directory
from .tangents import corner_frames, corner_normals
from .textures import metallic_roughness_pixels, normal_pixels, pixels, png_bytes

_Y_UP = np.array(((1, 0, 0), (0, 0, 1), (0, -1, 0)))
_FLOAT, _UNSIGNED_SHORT, _UNSIGNED_INT = 5126, 5123, 5125
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963


def _indexed_vertices(mesh, part):
    has_uv = part.texture is not None
    has_normal_map = has_uv and part.texture.pbr is not None
    if has_uv:
        triangles, normals, uv, tangents = corner_frames(mesh, part)
    else:
        triangles, normals = corner_normals(mesh)

    # Weld only corners that belong to the same geometric vertex and agree on
    # every exported attribute. Hard edges and opposing tangent signs stay split.
    width = 4 + 2 * has_uv + 4 * has_normal_map
    corners = np.empty((triangles.size, width), dtype="<f4")
    # Our build budget limits indices to well below the exact integer range of float32.
    corners[:, 0] = triangles.reshape(-1)
    corners[:, 1:4] = normals.reshape(-1, 3)
    if has_uv:
        corners[:, 4:6] = uv.reshape(-1, 2)
    if has_normal_map:
        corners[:, 6:10] = tangents.reshape(-1, 4)
    _, first, inverse = np.unique(corners, axis=0, return_index=True, return_inverse=True)
    # Vertex order follows first use by the index buffer.
    order = np.argsort(first)
    first = first[order]
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    # glTF reserves the all-ones index for primitive restart, including 65535.
    indices = remap[inverse.reshape(-1)].astype("<u2" if len(first) <= 65535 else "<u4")
    corners = corners[first]

    vertices = np.empty((len(first), width + 2), dtype="<f4")
    vertices[:, 0:3] = mesh.vertices[corners[:, 0].astype(np.intp)] @ _Y_UP.T
    vertices[:, 3:6] = corners[:, 1:4] @ _Y_UP.T
    fields = [("POSITION", 3), ("NORMAL", 3)]
    if has_uv:
        vertices[:, 6] = corners[:, 4]
        vertices[:, 7] = 1 - corners[:, 5]  # glTF locates (0, 0) at the upper-left texel.
        fields.append(("TEXCOORD_0", 2))
    if has_normal_map:
        vertices[:, 8:11] = corners[:, 6:9] @ _Y_UP.T
        # Positive green still points toward the top of the PNG. The proper
        # axis rotation preserves the authored bitangent and its handedness.
        vertices[:, 11] = corners[:, 9]
        fields.append(("TANGENT", 4))
    return vertices, fields, indices


class _Asset:
    def __init__(self, binary):
        self.binary = binary
        self.doc = {
            "asset": {"version": "2.0", "generator": f"Xtrifid {__version__}"},
            "scene": 0,
            "scenes": [{"nodes": []}],
            "nodes": [],
            "meshes": [],
            "materials": [],
            "bufferViews": [],
            "accessors": [],
        }
        self.mesh_indices = {}
        self.material_indices = {}
        self.texture_indices = {}

    def view(self, data, **properties):
        stream = self.binary
        stream.write(b"\0" * (-stream.tell() % 4))
        start = stream.tell()
        stream.write(data)
        views = self.doc["bufferViews"]
        views.append(
            {
                "buffer": 0,
                "byteOffset": start,
                "byteLength": stream.tell() - start,
                **properties,
            }
        )
        return len(views) - 1

    def accessor(self, view, component, count, kind, **properties):
        accessors = self.doc["accessors"]
        accessors.append(
            {
                "bufferView": view,
                "componentType": component,
                "count": count,
                "type": kind,
                **properties,
            }
        )
        return len(accessors) - 1

    def texture(self, role, image, name):
        encoded = png_bytes(image)
        key = (role, hashlib.sha256(encoded).digest())
        if key not in self.texture_indices:
            images = self.doc.setdefault("images", [])
            textures = self.doc.setdefault("textures", [])
            images.append(
                {
                    "name": f"{name}_{role}",
                    "mimeType": "image/png",
                    "bufferView": self.view(encoded),
                }
            )
            self.doc.setdefault("samplers", [{"magFilter": 9729, "minFilter": 9987}])
            self.texture_indices[key] = len(textures)
            textures.append({"source": len(images) - 1, "sampler": 0})
        return {"index": self.texture_indices[key]}

    def material(self, spec):
        if spec in self.material_indices:
            return self.material_indices[spec]
        index = len(self.doc["materials"])
        name = f"{spec.material}_{index + 1:02}" if spec else "Clay"
        pbr = {"metallicFactor": 0.0, "roughnessFactor": 1.0}
        material = {"name": name, "pbrMetallicRoughness": pbr}
        if spec is None:
            pbr["baseColorFactor"] = [0.60, 0.62, 0.69, 1.0]
        else:
            pbr["baseColorTexture"] = self.texture("BaseColor", pixels(spec), name)
            if spec.pbr is not None:
                material["normalTexture"] = self.texture("NormalGL", normal_pixels(spec), name)
                pbr["metallicFactor"] = 1.0
                pbr["metallicRoughnessTexture"] = self.texture(
                    "MetallicRoughness", metallic_roughness_pixels(spec), name
                )
        self.doc["materials"].append(material)
        self.material_indices[spec] = index
        return index

    def mesh(self, part, detail, style, name):
        if part in self.mesh_indices:
            return self.mesh_indices[part]
        material = self.material(part.texture)
        vertices, fields, indices = _indexed_vertices(build(part, "game", detail, style), part)
        view = self.view(vertices, target=_ARRAY_BUFFER, byteStride=vertices.strides[0])
        attributes = {}
        column = 0
        for semantic, width in fields:
            properties = {"byteOffset": column * 4}
            if semantic == "POSITION":
                properties["min"] = vertices[:, :3].min(axis=0).tolist()
                properties["max"] = vertices[:, :3].max(axis=0).tolist()
            attributes[semantic] = self.accessor(
                view, _FLOAT, len(vertices), f"VEC{width}", **properties
            )
            column += width
        view = self.view(indices, target=_ELEMENT_ARRAY_BUFFER)
        component = _UNSIGNED_SHORT if indices.itemsize == 2 else _UNSIGNED_INT
        primitive = {
            "attributes": attributes,
            "indices": self.accessor(view, component, len(indices), "SCALAR"),
            "material": material,
        }
        index = len(self.doc["meshes"])
        self.doc["meshes"].append({"name": name, "primitives": [primitive]})
        self.mesh_indices[part] = index
        return index

    def add_part(self, part, scene):
        # Keep the editable pivot and transform on the node. Equal parts with
        # different names/poses can reference a single local mesh and material.
        local = replace(part, name="Mesh", center=(0, 0, 0), rotation=(0, 0, 0))
        node = {
            "name": part.name,
            "mesh": self.mesh(local, scene.detail, scene.style, part.name),
        }
        if any(part.rotation):
            transform = np.eye(4)
            transform[:3, :3] = _Y_UP @ rotation_matrix(part.rotation) @ _Y_UP.T
            transform[:3, 3] = _Y_UP @ part.center
            node["matrix"] = transform.T.flatten().tolist()  # glTF matrices are column-major.
        elif any(part.center):
            node["translation"] = (_Y_UP @ part.center).tolist()
        self.doc["scenes"][0]["nodes"].append(len(self.doc["nodes"]))
        self.doc["nodes"].append(node)

    def write(self, output):
        length = self.binary.tell()
        self.doc["buffers"] = [{"byteLength": length}]
        metadata = json.dumps(
            self.doc, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        metadata += b" " * (-len(metadata) % 4)
        padding = b"\0" * (-length % 4)
        total = 28 + len(metadata) + length + len(padding)
        if total > 0xFFFFFFFF:
            raise ValueError("Превышен максимальный размер файла GLB")
        output.write(struct.pack("<4s3I4s", b"glTF", 2, total, len(metadata), b"JSON"))
        output.write(metadata)
        output.write(struct.pack("<I4s", length + len(padding), b"BIN\0"))
        self.binary.seek(0)
        shutil.copyfileobj(self.binary, output, length=1024 * 1024)
        output.write(padding)


def export_glb(path, scene, mode="game"):
    """Publish one complete game asset atomically, with at most one part and map in staging memory."""
    if mode != "game":
        raise ValueError("Для экспорта GLB выберите профиль Game Ready")
    if not isinstance(scene, Scene) or not 1 <= len(scene.parts) <= MAX_PARTS:
        raise ValueError("Выберите сцену для экспорта GLB")
    check_scene_budget(scene, mode)
    destination = Path(path)
    with tempfile.TemporaryFile(dir=destination.parent) as binary:
        asset = _Asset(binary)
        for part in scene.parts:
            asset.add_part(part, scene)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            try:
                asset.write(output)
                output.flush()
                os.fsync(output.fileno())
                output.close()
                os.replace(output.name, destination)
            finally:
                Path(output.name).unlink(missing_ok=True)
    _sync_directory(destination.parent)
