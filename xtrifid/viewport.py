"""GTK OpenGL viewport. Display lighting and grid are never written to the asset."""

import ctypes
from math import asin, atan2, cos, sin

import numpy as np
from OpenGL import GL
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.raw.GL.VERSION.GL_2_0 import glVertexAttribPointer

from xtrifid.gtk import Gdk, GLib, Gtk

from .geometry import rotation_matrix, triangles
from .textures import pixels, uv_layout

VERT = """#version 330 core
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in float part;
layout(location=3) in vec2 uv;
out vec2 texcoord;
uniform mat4 camera;
out vec3 n;
flat out int id;
void main(){ gl_Position=camera*vec4(position,1); n=normal; id=int(part); texcoord=uv; }"""
FRAG = """#version 330 core
in vec3 n;
flat in int id;
uniform int selected;
uniform int portrait;
uniform int mapped;
uniform sampler2D diffuse;
in vec2 texcoord;
uniform int passKind;
out vec4 color;
void main(){
    if(passKind==3){color=vec4(1.,.80,.42,1);return;}
    if(passKind==2){color=vec4(.18,.21,.26,1);return;}
    if(passKind==1){color=vec4(.13,.17,.22,1);return;}
    vec3 normal=normalize(n);
    float key=max(dot(normal,normalize(vec3(-.4,-.6,1.))),0.);
    float fill=max(dot(normal,normalize(vec3(.6,.7,.3))),0.);
    float light=.43+.38*key+.19*fill;
    vec3 base=id==selected?(portrait==1?vec3(.84,.82,.77):vec3(.52,.86,.82)):vec3(.77,.80,.88);
    if(mapped!=0) base=texture(diffuse,texcoord).rgb;
    color=vec4(base*light,1);
}"""


def _surface(vertices, mesh, index, part):
    corners = np.array(list(triangles(mesh.faces)))
    if not len(corners):
        return np.empty((0, 9), np.float32)
    points = vertices[corners]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    smooth = np.repeat(mesh.smooth, [len(face) - 2 for face in mesh.faces])
    averaged = np.zeros_like(vertices)
    for col in range(3):
        np.add.at(averaged, corners[smooth, col], normals[smooth])
    ns = np.repeat(normals[:, None, :], 3, axis=1)
    ns[smooth] = averaged[corners[smooth]]
    ns /= np.maximum(np.linalg.norm(ns, axis=2, keepdims=True), 1e-30)
    surface = np.zeros((points.size // 3, 9), np.float32)
    surface[:, :3], surface[:, 3:6], surface[:, 6] = (
        points.reshape(-1, 3),
        ns.reshape(-1, 3),
        index,
    )
    if part and part.texture is not None:
        uv, faces = uv_layout(mesh, part)
        uv_indices = np.asarray(list(triangles(faces))).reshape(-1)
        surface[:, 7:] = uv[uv_indices]
    return surface


def batches(meshes, parts, center, radius):
    surfaces, lines, owners, ranges = [], [], [], []
    surface_offset = line_offset = 0
    for index, mesh in enumerate(meshes):
        part = parts[index] if index < len(parts) else None
        vertices = (mesh.vertices - center) / radius
        surface = _surface(vertices, mesh, index, part)
        surfaces.append(surface)
        owners.extend([index] * (len(surface) // 3))
        edges = {
            tuple(sorted((a, b)))
            for face in mesh.faces
            for a, b in zip(face, face[1:] + face[:1])
        }
        pts = vertices[np.array(sorted(edges))].reshape(-1, 3)
        wire = np.zeros((len(pts), 9), np.float32)
        wire[:, :3] = pts
        lines.append(wire)
        ranges.append(
            (
                surface_offset,
                len(surface),
                line_offset,
                len(wire),
                part.texture if part else None,
            )
        )
        surface_offset += len(surface)
        line_offset += len(wire)
    surface = np.concatenate(surfaces) if surfaces else np.empty((0, 9), np.float32)
    wire = np.concatenate(lines) if lines else np.empty((0, 9), np.float32)
    return surface, wire, np.array(owners), ranges


class Viewport(Gtk.GLArea):
    def __init__(self, on_pick, on_error):
        super().__init__(hexpand=True, vexpand=True, focusable=True)
        self.set_allowed_apis(Gdk.GLAPI.GL)
        self.set_required_version(3, 3)
        self.set_has_depth_buffer(True)
        self.on_pick, self.on_error = on_pick, on_error
        self.update_property([Gtk.AccessibleProperty.LABEL], ["Предпросмотр 3D-модели"])
        self.selected = -1
        self.wire = False
        self.problem = None
        self.drawn = False
        self.program = None
        self.guide = np.empty((0, 9), dtype=np.float32)
        self.textures = {}
        self.guide_capacity = 0
        self.guide_dirty = True
        self.set_meshes([])
        self.reset_camera()
        self.connect("render", self.render_mesh)
        self.connect("unrealize", self.release_gl)
        drag = Gtk.GestureDrag()
        drag.connect("drag-begin", self.begin)
        drag.connect("drag-update", self.orbit)
        drag.connect("drag-end", self.end)
        self.add_controller(drag)
        scroll = Gtk.EventControllerScroll.new(Gtk.EventControllerScrollFlags.VERTICAL)
        scroll.connect(
            "scroll", lambda _, dx, dy: self.zoom_by(1.12 ** (-max(-10, min(10, dy))))
        )
        self.add_controller(scroll)
        keys = Gtk.EventControllerKey()
        keys.connect("key-pressed", self.key)
        self.add_controller(keys)

    def set_meshes(self, meshes, parts=()):
        if meshes:
            points = np.concatenate([mesh.vertices for mesh in meshes])
            lo, hi = points.min(axis=0), points.max(axis=0)
            center = (lo + hi) / 2
            radius = np.linalg.norm(hi - lo) / 2
            floor = (lo[2] - center[2]) / radius - 0.006
        else:
            center = np.zeros(3)
            radius = 1
            floor = -0.5
        frames = []
        for mesh, part in zip(meshes, parts):
            vertices = mesh.vertices
            direction = None
            if part.shape == "human":
                axes = rotation_matrix(part.rotation)
                height = (vertices - part.center) @ axes[:, 2]
                vertices = vertices[height >= height.max() - 0.145 * np.ptp(height)]
                direction = -axes[:, 1]
            lo, hi = vertices.min(axis=0), vertices.max(axis=0)
            frames.append(
                (
                    ((hi + lo) / 2 - center) / radius,
                    np.linalg.norm(hi - lo) / (2 * radius),
                    direction,
                )
            )
        surface, wire, owners, ranges = batches(meshes, parts, center, radius)
        desired_textures = tuple(dict.fromkeys(spec for *_, spec in ranges if spec))
        grid = []
        for i in range(-12, 13):
            t = i / 5
            grid.extend(((t, -2.4, floor), (t, 2.4, floor), (-2.4, t, floor), (2.4, t, floor)))
        floor_lines = np.column_stack((grid, np.zeros((len(grid), 6))))
        buffer = np.concatenate((surface, wire, floor_lines), dtype=np.float32)
        # Keep the pick coordinates as a view of the same storage sent to OpenGL.
        pick_points = buffer[: len(surface), :3].reshape(-1, 3, 3)
        self.center, self.radius = center, radius
        self.frames = frames
        self.counts = len(surface), len(wire), len(grid)
        self.buffer, self.owners, self.pick_points = buffer, owners, pick_points
        self.ranges, self.desired_textures = ranges, desired_textures
        self.sim_surface = self.sim_replaces = None
        if getattr(self, "focused", None) is not None:
            if self.focused < len(frames):
                self.target = frames[self.focused][0]
            else:
                self.reset_camera()
        self.dirty = True
        self.queue_render()

    def set_simulation_preview(self, mesh, replace_selected=False):
        """Stream one transient mesh while keeping the static buffers and picking valid."""
        self.sim_replaces = self.selected if mesh is not None and replace_selected else None
        self.sim_surface = (
            None
            if mesh is None
            else _surface(
                (mesh.vertices - self.center) / self.radius, mesh, self.selected, None
            )
        )
        self.sim_material = None
        if self.sim_replaces is not None and 0 <= self.sim_replaces < len(self.ranges):
            start, count, _, _, material = self.ranges[self.sim_replaces]
            # Replacement simulations retain the selected garment's corner order.
            # Keep rest-space UVs attached to the cloth as its positions change.
            if material and count == len(self.sim_surface):
                self.sim_material = material
                self.sim_surface[:, 7:] = self.buffer[start : start + count, 7:]
        self.sim_dirty = True
        self.queue_render()

    def reset_camera(self):
        self.focused = None
        self.target = np.zeros(3)
        figure = next(
            (direction for _, _, direction in self.frames if direction is not None), None
        )
        self.yaw, self.pitch, self.zoom = 0.65, 0.42, 1.0
        if figure is not None:
            self.yaw = atan2(figure[0], -figure[1]) + 0.14
            self.pitch = 0.06
        self.queue_render()

    def focus_selected(self):
        if not 0 <= self.selected < len(self.frames):
            return False
        self.focused = self.selected
        self.target, extent, direction = self.frames[self.selected]
        self.zoom = min(32, 1 / max(0.001, extent))
        if direction is not None:
            self.yaw = atan2(direction[0], -direction[1])
            self.pitch = asin(np.clip(direction[2], -1, 1))
        self.queue_render()
        return True

    def set_guide(self, points=()):
        self.guide = np.zeros((len(points), 9), dtype=np.float32)
        if len(points):
            self.guide[:, :3] = (np.asarray(points) - self.center) / self.radius
        if len(points) > self.guide_capacity:
            self.guide_capacity = len(points)
            self.dirty = True
        self.guide_dirty = True
        self.queue_render()

    def matrix(self):
        yaw, pitch = self.yaw, self.pitch
        right = np.array([cos(yaw), sin(yaw), 0])
        back = np.array([sin(yaw) * cos(pitch), -cos(yaw) * cos(pitch), sin(pitch)])
        up = np.cross(back, right)
        aspect = max(self.get_width(), 1) / max(1, self.get_height())
        scale = self.zoom * 0.89
        return np.array(
            [right * scale / max(1, aspect), up * scale * min(1, aspect), -back * 0.24]
        )

    def screen_position(self, world):
        x, y, _ = self.matrix() @ (
            (np.asarray(world) - self.center) / self.radius - self.target
        )
        return (x + 1) * self.get_width() / 2, (1 - y) * self.get_height() / 2

    def pick(self, x, y):
        if not len(self.owners):
            return None
        p = np.array([2 * x / self.get_width() - 1, 1 - 2 * y / self.get_height()])
        matrix = self.matrix()
        a, b, c = (self.pick_points @ matrix.T - matrix @ self.target).transpose(1, 0, 2)
        ab, ac = b - a, c - a
        d = p - a[:, :2]
        det = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
        usable = np.abs(det) > 1e-14
        den = np.where(usable, det, 1)
        u = (d[:, 0] * ac[:, 1] - d[:, 1] * ac[:, 0]) / den
        v = (ab[:, 0] * d[:, 1] - ab[:, 1] * d[:, 0]) / den
        inside = usable & (u >= 0) & (v >= 0) & (u + v <= 1)
        if inside.any():
            depth = np.where(inside, a[:, 2] + u * ab[:, 2] + v * ac[:, 2], np.inf)
            return int(self.owners[depth.argmin()])
        return None

    def begin(self, gesture, x, y):
        self.grab_focus()
        self.start = (x, y, self.yaw, self.pitch)

    def orbit(self, gesture, dx, dy):
        if abs(dx) + abs(dy) > 4:
            self.yaw = self.start[2] - dx * 0.009
            self.pitch = max(-1.5, min(1.5, self.start[3] + dy * 0.009))
            self.queue_render()

    def end(self, gesture, dx, dy):
        if abs(dx) + abs(dy) <= 4:
            which = self.pick(*self.start[:2])
            if which is not None:
                self.on_pick(which)

    def zoom_by(self, factor):
        self.zoom = min(32, max(0.2, self.zoom * factor))
        self.queue_render()
        return True

    def key(self, controller, key, code, state):
        if key in (Gdk.KEY_plus, Gdk.KEY_equal, Gdk.KEY_minus):
            return self.zoom_by(0.85 if key == Gdk.KEY_minus else 1.15)
        if key == Gdk.KEY_Home:
            self.reset_camera()
        elif key in (Gdk.KEY_f, Gdk.KEY_F):
            return self.focus_selected()
        elif key in (Gdk.KEY_Left, Gdk.KEY_Right):
            self.yaw += 0.12 if key == Gdk.KEY_Right else -0.12
        elif key in (Gdk.KEY_Up, Gdk.KEY_Down):
            self.pitch = max(
                -1.5, min(1.5, self.pitch + (0.12 if key == Gdk.KEY_Up else -0.12))
            )
        else:
            return False
        self.queue_render()
        return True

    def init_gl(self):
        self.program = compileProgram(
            compileShader(VERT, GL.GL_VERTEX_SHADER), compileShader(FRAG, GL.GL_FRAGMENT_SHADER)
        )
        self.vao, self.sim_vao = GL.glGenVertexArrays(2)
        self.vbo, self.sim_vbo = GL.glGenBuffers(2)
        for vao, vbo in ((self.vao, self.vbo), (self.sim_vao, self.sim_vbo)):
            GL.glBindVertexArray(vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
            for i, size, offset in ((0, 3, 0), (1, 3, 12), (2, 1, 24), (3, 2, 28)):
                GL.glEnableVertexAttribArray(i)
                glVertexAttribPointer(i, size, GL.GL_FLOAT, False, 36, ctypes.c_void_p(offset))
        self.uniforms = {
            name: GL.glGetUniformLocation(self.program, name)
            for name in ("camera", "passKind", "selected", "portrait", "mapped", "diffuse")
        }
        self.dirty = self.sim_dirty = True
        self.sim_capacity = 0

    def release_gl(self, area):
        self.make_current()
        if self.program and not self.get_error():
            if self.textures:
                GL.glDeleteTextures(list(self.textures.values()))
            GL.glDeleteVertexArrays(2, [self.vao, self.sim_vao])
            GL.glDeleteBuffers(2, [self.vbo, self.sim_vbo])
            GL.glDeleteProgram(self.program)
        self.textures.clear()
        self.program = None

    def _upload_textures(self):
        for spec in self.textures.keys() - set(self.desired_textures):
            GL.glDeleteTextures([self.textures.pop(spec)])
        missing = [s for s in self.desired_textures if s not in self.textures]
        if not missing:
            return
        # At most one image per frame when reopening a many-material project.
        spec = missing[0]
        image = np.ascontiguousarray(pixels(spec)[::-1])
        name = GL.glGenTextures(1)
        try:
            GL.glBindTexture(GL.GL_TEXTURE_2D, name)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR
            )
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGB8,
                spec.resolution,
                spec.resolution,
                0,
                GL.GL_RGB,
                GL.GL_UNSIGNED_BYTE,
                image,
            )
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        except Exception:
            GL.glDeleteTextures([name])
            raise
        self.textures[spec] = name
        if len(missing) > 1:
            GLib.idle_add(self.queue_render)

    def render_mesh(self, area, context):
        try:
            if not self.program:
                self.init_gl()
            GL.glActiveTexture(GL.GL_TEXTURE0)
            self._upload_textures()
            GL.glBindVertexArray(self.vao)
            if self.dirty:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
                GL.glBufferData(
                    GL.GL_ARRAY_BUFFER,
                    self.buffer.nbytes + self.guide_capacity * self.buffer.strides[0],
                    None,
                    GL.GL_DYNAMIC_DRAW,
                )
                GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, self.buffer.nbytes, self.buffer)
                self.dirty = False
                self.guide_dirty = True
            if self.guide_dirty and len(self.guide):
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
                GL.glBufferSubData(
                    GL.GL_ARRAY_BUFFER, self.buffer.nbytes, self.guide.nbytes, self.guide
                )
                self.guide_dirty = False
            matrix = np.eye(4, dtype=np.float32)
            matrix[:3, :3] = self.matrix()
            matrix[:3, 3] = -matrix[:3, :3] @ self.target
            GL.glUseProgram(self.program)
            GL.glUniformMatrix4fv(self.uniforms["camera"], 1, True, matrix)
            GL.glUniform1i(self.uniforms["selected"], self.selected)
            human = (
                0 <= self.selected < len(self.frames)
                and self.frames[self.selected][2] is not None
            )
            GL.glUniform1i(self.uniforms["portrait"], int(human))
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glClearColor(0.068, 0.081, 0.105, 1)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            faces, edges, grid = self.counts
            GL.glUniform1i(self.uniforms["passKind"], 2)
            GL.glDrawArrays(GL.GL_LINES, faces + edges, grid)
            GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
            GL.glPolygonOffset(1, 1)
            GL.glUniform1i(self.uniforms["passKind"], 0)
            GL.glUniform1i(self.uniforms["diffuse"], 0)
            for index, (start, count, _, _, spec) in enumerate(self.ranges):
                if index == self.sim_replaces:
                    continue
                name = self.textures.get(spec, 0)
                GL.glUniform1i(self.uniforms["mapped"], int(bool(name)))
                GL.glBindTexture(GL.GL_TEXTURE_2D, name)
                GL.glDrawArrays(GL.GL_TRIANGLES, start, count)
            if self.sim_surface is not None and len(self.sim_surface):
                name = self.textures.get(self.sim_material, 0)
                GL.glUniform1i(self.uniforms["mapped"], int(bool(name)))
                GL.glBindTexture(GL.GL_TEXTURE_2D, name)
                GL.glBindVertexArray(self.sim_vao)
                if self.sim_dirty:
                    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.sim_vbo)
                    if self.sim_surface.nbytes > self.sim_capacity:
                        self.sim_capacity = self.sim_surface.nbytes
                        GL.glBufferData(
                            GL.GL_ARRAY_BUFFER, self.sim_capacity, None, GL.GL_STREAM_DRAW
                        )
                    GL.glBufferSubData(
                        GL.GL_ARRAY_BUFFER, 0, self.sim_surface.nbytes, self.sim_surface
                    )
                    self.sim_dirty = False
                GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(self.sim_surface))
                GL.glBindVertexArray(self.vao)
            GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
            if self.wire:
                GL.glUniform1i(self.uniforms["passKind"], 1)
                for index, (_, _, start, count, _) in enumerate(self.ranges):
                    if index != self.sim_replaces:
                        GL.glDrawArrays(GL.GL_LINES, faces + start, count)
            if len(self.guide):
                GL.glDepthFunc(GL.GL_LEQUAL)
                GL.glUniform1i(self.uniforms["passKind"], 3)
                GL.glDrawArrays(GL.GL_LINE_STRIP, len(self.buffer), len(self.guide))
                if len(self.guide) <= 2:
                    GL.glPointSize(6)
                    GL.glDrawArrays(GL.GL_POINTS, len(self.buffer), len(self.guide))
                GL.glDepthFunc(GL.GL_LESS)
            self.drawn = True
        except Exception as exc:
            if self.problem is None:
                self.problem = str(exc)
                self.on_error(f"Ошибка OpenGL: {exc}")
        return True
