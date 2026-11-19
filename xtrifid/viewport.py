"""GTK OpenGL viewport. Display lighting and grid are never written to the asset."""

import ctypes
from math import cos, sin

import numpy as np
from OpenGL import GL
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.raw.GL.VERSION.GL_2_0 import glVertexAttribPointer

from xtrifid.gtk import Gdk, Gtk

from .geometry import triangles

VERT = """#version 330 core
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in float part;
uniform mat4 camera;
out vec3 n;
flat out int id;
void main(){ gl_Position=camera*vec4(position,1); n=normal; id=int(part); }"""
FRAG = """#version 330 core
in vec3 n;
flat in int id;
uniform int selected;
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
    vec3 base=id==selected?vec3(.52,.86,.82):vec3(.77,.80,.88);
    color=vec4(base*light,1);
}"""


def batches(meshes, center, radius):
    surfaces, lines, owners = [], [], []
    for index, mesh in enumerate(meshes):
        vertices = (mesh.vertices - center) / radius
        corners = np.array(list(triangles(mesh.faces)))
        points = vertices[corners]
        normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
        smooth = np.repeat(mesh.smooth, [len(face) - 2 for face in mesh.faces])
        averaged = np.zeros_like(vertices)
        for col in range(3):
            np.add.at(averaged, corners[smooth, col], normals[smooth])
        ns = np.repeat(normals[:, None, :], 3, axis=1)
        ns[smooth] = averaged[corners[smooth]]
        ns /= np.maximum(np.linalg.norm(ns, axis=2, keepdims=True), 1e-30)
        surfaces.append(
            np.column_stack(
                (points.reshape(-1, 3), ns.reshape(-1, 3), np.full(points.size // 3, index))
            )
        )
        owners.extend([index] * len(points))
        edges = {
            tuple(sorted((a, b)))
            for face in mesh.faces
            for a, b in zip(face, face[1:] + face[:1])
        }
        pts = vertices[np.array(sorted(edges))].reshape(-1, 3)
        lines.append(np.column_stack((pts, np.zeros((len(pts), 4)))))
    surface = np.concatenate(surfaces) if surfaces else np.empty((0, 7))
    wire = np.concatenate(lines) if lines else np.empty((0, 7))
    return surface, wire, np.array(owners)


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
        self.guide = np.empty((0, 7), dtype=np.float32)
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

    def set_meshes(self, meshes):
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
        surface, wire, owners = batches(meshes, center, radius)
        grid = []
        for i in range(-12, 13):
            t = i / 5
            grid.extend(((t, -2.4, floor), (t, 2.4, floor), (-2.4, t, floor), (2.4, t, floor)))
        floor_lines = np.column_stack((grid, np.zeros((len(grid), 4))))
        buffer = np.concatenate((surface, wire, floor_lines), dtype=np.float32)
        # Keep the pick coordinates as a view of the same storage sent to OpenGL.
        pick_points = buffer[: len(surface), :3].reshape(-1, 3, 3)
        self.center, self.radius = center, radius
        self.counts = len(surface), len(wire), len(grid)
        self.buffer, self.owners, self.pick_points = buffer, owners, pick_points
        self.dirty = True
        self.queue_render()

    def reset_camera(self):
        self.yaw, self.pitch, self.zoom = 0.65, 0.42, 1.0
        self.queue_render()

    def set_guide(self, points=()):
        self.guide = np.zeros((len(points), 7), dtype=np.float32)
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
        x, y, _ = self.matrix() @ ((np.asarray(world) - self.center) / self.radius)
        return (x + 1) * self.get_width() / 2, (1 - y) * self.get_height() / 2

    def pick(self, x, y):
        if not len(self.owners):
            return None
        p = np.array([2 * x / self.get_width() - 1, 1 - 2 * y / self.get_height()])
        a, b, c = (self.pick_points @ self.matrix().T).transpose(1, 0, 2)
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
        self.zoom = min(8, max(0.2, self.zoom * factor))
        self.queue_render()
        return True

    def key(self, controller, key, code, state):
        if key in (Gdk.KEY_plus, Gdk.KEY_equal, Gdk.KEY_minus):
            return self.zoom_by(0.85 if key == Gdk.KEY_minus else 1.15)
        if key == Gdk.KEY_Home:
            self.reset_camera()
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
        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        for i, size, offset in ((0, 3, 0), (1, 3, 12), (2, 1, 24)):
            GL.glEnableVertexAttribArray(i)
            glVertexAttribPointer(i, size, GL.GL_FLOAT, False, 28, ctypes.c_void_p(offset))
        self.uniforms = {
            name: GL.glGetUniformLocation(self.program, name)
            for name in ("camera", "passKind", "selected")
        }
        self.dirty = True

    def release_gl(self, area):
        self.make_current()
        if self.program and not self.get_error():
            GL.glDeleteVertexArrays(1, [self.vao])
            GL.glDeleteBuffers(1, [self.vbo])
            GL.glDeleteProgram(self.program)
        self.program = None

    def render_mesh(self, area, context):
        try:
            if not self.program:
                self.init_gl()
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
            GL.glUseProgram(self.program)
            GL.glUniformMatrix4fv(self.uniforms["camera"], 1, True, matrix)
            GL.glUniform1i(self.uniforms["selected"], self.selected)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glClearColor(0.068, 0.081, 0.105, 1)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            faces, edges, grid = self.counts
            GL.glUniform1i(self.uniforms["passKind"], 2)
            GL.glDrawArrays(GL.GL_LINES, faces + edges, grid)
            GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
            GL.glPolygonOffset(1, 1)
            GL.glUniform1i(self.uniforms["passKind"], 0)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, faces)
            GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
            if self.wire:
                GL.glUniform1i(self.uniforms["passKind"], 1)
                GL.glDrawArrays(GL.GL_LINES, faces, edges)
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
