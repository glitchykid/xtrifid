"""Fixed-step geometry simulation. Coordinates, forces and elapsed time use SI units.

Cloth and hair use position-based distance constraints (Müller et al., 2006).
The water basin solves the linearised shallow-water equations on a staggered
height/velocity grid with impermeable walls and a Courant-limited substep.
"""

from math import ceil, pi

import numpy as np

from .geometry import Mesh, rotation_matrix
from .model import DETAILS
from .organic import CLOAK
from .resources import mesh_triangle_budget
from .simulation_fluid import TRIANGLE_LIMITS, ParticleFluid
from .textures import texture_working_bytes

FPS = 30
MAX_FRAME = 180
KINDS = ("cloth", "hair", "water", "fluid")
_NAMES = {
    "cloth": "Ткань",
    "hair": "Волосы",
    "water": "Водная поверхность",
    "fluid": "Объёмная жидкость",
}


def _grid_faces(rows, cols):
    grid = np.arange(rows * cols).reshape(rows, cols)
    return [
        tuple(f)
        for f in np.stack((grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]), axis=-1)
        .reshape(-1, 4)
        .tolist()
    ]


def _closed_sheet(rows, cols):
    top = _grid_faces(rows, cols)
    k = rows * cols
    grid = np.arange(k).reshape(rows, cols)
    rim = np.concatenate((grid[0], grid[1:, -1], grid[-1, -2::-1], grid[-2:0:-1, 0])).tolist()
    walls = [(b, a, a + k, b + k) for a, b in zip(rim, rim[1:] + rim[:1])]
    return tuple(top + [tuple(i + k for i in reversed(f)) for f in top] + walls)


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-16)


def _basis(grid):
    along = _unit(np.gradient(grid, axis=1))
    normal = _unit(np.cross(along, np.gradient(grid, axis=0)))
    return np.stack((along, np.cross(normal, along), normal), axis=-1)


def _frame(vertices, faces, smooth):
    vertices.setflags(write=False)
    return Mesh(vertices, faces, smooth)


class _Obstacle:
    def __init__(self, center, radii, rotation, rounded):
        self.center = center
        self.radii = np.maximum(radii, 1e-7)
        self.rotation = rotation
        self.rounded = rounded

    def clearance(self, points):
        local = (points - self.center) @ self.rotation
        if self.rounded:
            return (np.linalg.norm(local / self.radii, axis=1) - 1) * self.radii.min()
        gap = abs(local) - self.radii
        return np.linalg.norm(np.maximum(gap, 0), axis=1) + np.minimum(np.max(gap, axis=1), 0)

    def project(self, points, margin):
        local = (points - self.center) @ self.rotation
        radii = self.radii + margin
        scaled = local / radii
        if self.rounded:
            norms = np.linalg.norm(scaled, axis=1)
            inside = norms < 1
            displaced = scaled[inside].copy()
            displaced[norms[inside] < 1e-10] = (0, 0, 1)
            local[inside] = _unit(displaced) * radii
        else:
            gaps = radii - abs(local)
            inside = (gaps > 0).all(axis=1)
            ids = np.flatnonzero(inside)
            axis = gaps[inside].argmin(axis=1)
            local[ids, axis] = np.where(local[ids, axis] < 0, -1, 1) * radii[axis]
        if inside.any():
            points[inside] = local[inside] @ self.rotation.T + self.center
        return inside


def _local_bounds(part, mesh):
    rotation = rotation_matrix(part.rotation)
    local = (mesh.vertices - part.center) @ rotation
    return rotation, local, local.min(axis=0), local.max(axis=0)


class _BodyObstacle:
    """Conservative angular cross sections measured on the actual posed skin."""

    def __init__(self, part, rotation, points):
        self.rotation = rotation
        self.origin = np.asarray(part.center)
        low, high = points.min(0), points.max(0)
        self.bottom, self.top = low[2], high[2]
        self.dz = (self.top - self.bottom) / 79
        self.da = 2 * pi / 64
        row = np.rint((points[:, 2] - self.bottom) / self.dz).astype(int)
        # Vertical sections follow the midpoint of the figure, so a walking
        # or scaled figure keeps its actual chest and head location.
        self.centers = np.zeros((80, 2))
        for axis in (0, 1):
            lower = np.full(80, np.inf)
            upper = np.full(80, -np.inf)
            np.minimum.at(lower, row, points[:, axis])
            np.maximum.at(upper, row, points[:, axis])
            good = np.isfinite(lower)
            self.centers[:, axis] = np.interp(
                np.arange(80), np.flatnonzero(good), (upper[good] + lower[good]) / 2
            )
        delta = points[:, :2] - self.centers[row]
        angle = np.arctan2(delta[:, 1], delta[:, 0])
        col = np.rint(angle / self.da).astype(int) % 64
        radii = np.zeros((80, 64))
        np.maximum.at(radii, (row, col), np.linalg.norm(delta, axis=1))
        for i, rad in enumerate(radii):
            good = np.flatnonzero(rad > 0)
            if len(good):
                radii[i] = np.interp(np.arange(64), good, rad[good], period=64)
        # A sparse LOD can miss a row. An angular and vertical one-cell
        # envelope also leaves clearance for a face between contact vertices.
        rows = np.flatnonzero(radii.max(1) > 0)
        for j in range(64):
            radii[:, j] = np.interp(np.arange(80), rows, radii[rows, j])
        radii = np.maximum.reduce(
            (radii, np.roll(radii, 1, axis=1), np.roll(radii, -1, axis=1))
        )
        padded = np.pad(radii, ((1, 1), (0, 0)), mode="edge")
        self.radii = np.maximum.reduce((radii, padded[:-2], padded[2:]))

    def _contact(self, points):
        local = (points - self.origin) @ self.rotation
        z = np.clip((local[:, 2] - self.bottom) / self.dz, 0, 79)
        r = np.minimum(z.astype(int), 78)
        w = z - r
        center = self.centers[r] * (1 - w[:, None]) + self.centers[r + 1] * w[:, None]
        delta = local[:, :2] - center
        dist = np.linalg.norm(delta, axis=1)
        radial = delta / np.maximum(dist[:, None], 1e-10)
        radial[dist < 1e-10] = (1, 0)
        angle = np.mod(np.arctan2(delta[:, 1], delta[:, 0]) / self.da, 64)
        c = angle.astype(int) % 64
        t = angle - np.floor(angle)
        c1 = (c + 1) % 64
        near = self.radii[r, c] * (1 - t) + self.radii[r, c1] * t
        far = self.radii[r + 1, c] * (1 - t) + self.radii[r + 1, c1] * t
        radius = near + (far - near) * w
        drift = np.einsum("ij,ij->i", radial, self.centers[r + 1] - self.centers[r])
        slope = (far - near + drift) / self.dz
        dtheta = (
            (self.radii[r, c1] - self.radii[r, c]) * (1 - w)
            + (self.radii[r + 1, c1] - self.radii[r + 1, c]) * w
        ) / self.da
        tangent = np.column_stack((-radial[:, 1], radial[:, 0]))
        normal = np.column_stack(
            (radial - tangent * (dtheta / np.maximum(radius, 1e-8))[:, None], -slope)
        )
        length = np.linalg.norm(normal, axis=1)
        clearance = (dist - radius) / length
        normal /= length[:, None]
        # Close the crown and the soles of the sampled profile.
        for cap, vector in (
            (local[:, 2] - self.top, (0, 0, 1)),
            (self.bottom - local[:, 2], (0, 0, -1)),
        ):
            use = cap > clearance
            clearance[use] = cap[use]
            normal[use] = vector
        return clearance, normal

    def project(self, points, margin):
        distance, normal = self._contact(points)
        inside = distance < margin
        points[inside] += (normal[inside] @ (self.rotation.T)) * (
            margin - distance[inside, None]
        )
        return inside

    def clearance(self, points):
        return self._contact(points)[0]


def _obstacles(part, mesh, scalp=False):
    rotation, v, lo, hi = _local_bounds(part, mesh)
    if part.shape != "human":
        return [
            _Obstacle(
                (hi + lo) / 2 @ rotation.T + part.center,
                (hi - lo) / 2,
                rotation,
                part.shape in ("sphere", "head"),
            )
        ]
    if scalp:
        # The smoothly rounded head contact supports gliding locks; the
        # rayed shoulder and trunk envelope is used for draped garments.
        skull = v[v[:, 2] >= lo[2] + 0.858 * (hi[2] - lo[2])]
        a, b = skull.min(axis=0), skull.max(axis=0)
        return [_Obstacle((a + b) / 2 @ rotation.T + part.center, (b - a) / 2, rotation, True)]
    return [_BodyObstacle(part, rotation, v)]


def _batches(grid, offsets):
    """Edge-colour each family so a Gauss-Seidel batch never writes a vertex twice."""
    rows, cols = grid.shape
    batches = []
    for dr, dc, stiffness in offsets:
        r, c = np.indices((rows - dr, cols - abs(dc)))
        if dc < 0:
            c -= dc
        phase = (r // dr if dr else c // abs(dc)) % 2
        for parity in range(2):
            chosen = phase == parity
            a, b = grid[r[chosen], c[chosen]], grid[r[chosen] + dr, c[chosen] + dc]
            if len(a):
                batches.append((a, b, stiffness))
    return batches


class _Particles:
    def initialize(self, grid, pinned, offsets, obstacles, floor, margin):
        self.grid_shape = grid.shape[:2]
        self.rest = grid.reshape(-1, 3).copy()
        self.pos = self.rest.copy()
        self.prev = self.rest.copy()
        self.pinned = np.asarray(pinned).ravel()
        self.obstacles = obstacles
        self.groom = None
        self.probes = None
        self.floor = floor
        self.margin = margin
        ids = np.arange(len(self.pos)).reshape(self.grid_shape)
        mass = (~self.pinned).astype(float)
        self.constraints = []
        for a, b, k in _batches(ids, offsets):
            total = np.maximum(1, mass[a] + mass[b])
            self.constraints.append(
                (
                    a,
                    b,
                    np.linalg.norm(self.rest[b] - self.rest[a], axis=1),
                    (k * mass[a] / total)[:, None],
                    (k * mass[b] / total)[:, None],
                )
            )

    def snapshot(self):
        return self.pos.copy(), self.prev.copy()

    def restore(self, state):
        self.pos[:], self.prev[:] = state

    def step(self, frame, wind, gravity):
        dt = 1 / (FPS * 2)
        for substep in range(2):
            t = (frame + substep / 2) / FPS
            acceleration = np.array(
                (wind * (0.78 + 0.22 * np.sin(4 * t)), wind * 0.24 * np.cos(2.5 * t), -gravity)
            )
            predicted = self.pos + (self.pos - self.prev) * 0.979 + acceleration * dt * dt
            if self.groom is not None:
                predicted += self.groom * (self.rest - predicted)
            self.prev, self.pos = self.pos, predicted
            self.pos[self.pinned] = self.rest[self.pinned]
            contact = np.zeros(len(self.pos), bool)
            for iteration in range(3):
                for a, b, rest, wa, wb in self.constraints:
                    delta = self.pos[b] - self.pos[a]
                    length = np.maximum(np.linalg.norm(delta, axis=1), 1e-14)
                    corr = delta * (1 - rest / length)[:, None]
                    self.pos[a] += wa * corr
                    self.pos[b] -= wb * corr
                for obstacle in self.obstacles:
                    contact |= obstacle.project(self.pos, self.margin)
                if iteration == 2 and self.probes is not None:
                    self.project_panels()
                self.pos[:, 2] = np.maximum(self.pos[:, 2], self.floor + self.margin)
                self.pos[self.pinned] = self.rest[self.pinned]
            # Dissipate sliding motion at solid contacts; folds over an edge
            # then settle instead of accumulating fictitious tangential speed.
            self.prev[contact] += 0.46 * (self.pos[contact] - self.prev[contact])


class _Cloth(_Particles):
    def __init__(self, part, mesh, level, obstacles):
        v = mesh.vertices
        lo, hi = v.min(0), v.max(0)
        width, depth, height = hi - lo
        margin = max(1e-6, min(width, depth, height) * 0.005)
        if part.shape == "cloak":
            rows = len(CLOAK)
            cols = len(v) // (2 * rows)
            if 2 * rows * cols != len(v) or not 3 <= cols <= 80:
                raise ValueError("Не удалось найти сетку полотна плаща")
            outer, inner = np.split(v, 2)
            grid = outer.reshape(rows, cols, 3)
            self.thickness = np.einsum(
                "...ji,...j->...i", _basis(grid), (inner - outer).reshape(grid.shape)
            )
            self.faces, self.smooth = mesh.faces, mesh.smooth
            pinned = np.zeros((rows, cols), bool)
            pinned[-1] = True
            # The tailored yoke rests on the shoulder girdle; pin the short
            # shoulder seams as well as the neckline, leaving the hem free.
            local = (grid[-2] - part.center) @ rotation_matrix(part.rotation)
            pinned[-2] = abs(local[:, 0]) > 0.87 * abs(local[:, 0]).max()
        else:
            rows, cols = ((9, 12), (13, 19), (17, 27))[level]
            t, u = np.meshgrid(np.linspace(0, 1, rows), np.linspace(0, 1, cols), indexing="ij")
            if part.shape == "human":
                rotation, _, a, b = _local_bounds(part, mesh)
                h = b[2] - a[2]
                angle = 0.32 + (2 * pi - 0.64) * u
                grid = (
                    np.stack(
                        (
                            h * (0.069 + 0.135 * t) * np.sin(angle),
                            h * (0.004 + 0.042 * t) - h * (0.061 + 0.095 * t) * np.cos(angle),
                            a[2] + h * (0.85 - 0.47 * t) + 0 * u,
                        ),
                        axis=-1,
                    )
                    @ rotation.T
                    + part.center
                )
            else:
                # Pin the back edge; the free edge starts beyond the support and can fall.
                grid = np.stack(
                    (
                        lo[0] + width * (0.5 + 1.18 * (u - 0.5)),
                        lo[1] + depth * (0.07 + 1.58 * t),
                        hi[2] + margin * 3 + np.zeros_like(t),
                    ),
                    axis=-1,
                )
            self.thickness = np.zeros_like(grid)
            self.thickness[:, :, 2] = -margin
            self.faces = _closed_sheet(rows, cols)
            self.smooth = (True,) * (2 * (rows - 1) * (cols - 1)) + (False,) * (
                2 * (rows + cols - 2)
            )
            pinned = np.zeros((rows, cols), bool)
            pinned[0] = True
        self.inner_at_pins = (
            grid + np.einsum("...ij,...j->...i", _basis(grid), self.thickness)
        )[pinned].copy()
        spacing = min(
            np.linalg.norm(np.diff(grid, axis=0), axis=2).mean(),
            np.linalg.norm(np.diff(grid, axis=1), axis=2).mean(),
        )
        self.initialize(
            grid,
            pinned,
            ((1, 0, 1), (0, 1, 1), (1, 1, 0.8), (1, -1, 0.8), (2, 0, 0.34), (0, 2, 0.34)),
            obstacles,
            min(0, lo[2]),
            max(2 * margin, 0.39 * spacing),
        )

        if obstacles:
            # Long, low-poly cloak faces can cut across a rounded hip while
            # both vertices stay outside. Enforce contacts on virtual points
            # without adding vertices or changing the garment topology.
            ids = np.arange(rows * cols).reshape(rows, cols)
            panels = np.stack(
                (ids[:-1, :-1], ids[1:, :-1], ids[1:, 1:], ids[:-1, 1:]), axis=-1
            ).reshape(-1, 4)
            vertical = np.stack((ids[:-1], ids[:-1], ids[1:], ids[1:]), axis=-1).reshape(-1, 4)
            horizontal = np.stack(
                (ids[:, :-1], ids[:, :-1], ids[:, 1:], ids[:, 1:]), axis=-1
            ).reshape(-1, 4)
            self.probes = np.concatenate((panels, vertical, horizontal))
            movable = (~self.pinned)[self.probes]
            self.probe_weights = movable / np.maximum(movable.mean(axis=1, keepdims=True), 0.25)

    def project_panels(self):
        middle = self.pos[self.probes].mean(axis=1)
        projected = middle.copy()
        for obstacle in self.obstacles:
            obstacle.project(projected, self.margin)
        moved = np.any(projected != middle, axis=1)
        if not moved.any():
            return
        delta = (projected - middle)[moved, None, :] * self.probe_weights[moved, :, None]
        ids = self.probes[moved].reshape(-1)
        total = np.maximum(1, np.bincount(ids, minlength=len(self.pos)))
        delta = delta.reshape(-1, 3)
        for axis in range(3):
            self.pos[:, axis] += (
                np.bincount(ids, weights=delta[:, axis], minlength=len(self.pos)) / total
            )

    def mesh(self):
        grid = self.pos.reshape(*self.grid_shape, 3)
        offsets = np.einsum("...ij,...j->...i", _basis(grid), self.thickness)
        inner = (grid + offsets).reshape(-1, 3)
        inner[self.pinned] = self.inner_at_pins
        return _frame(np.concatenate((self.pos, inner)), self.faces, self.smooth)


def _scalp(head, center, radius, rotation, origin, level):
    # A closed underlayer follows the actual crown between the moving locks.
    # The hairline rises at the forehead and lowers around the nape and temples.
    rings, sides = ((4, 20), (6, 28), (8, 36))[level]
    theta = 2 * pi * np.arange(sides) / sides
    front = np.maximum(0, -np.sin(theta))
    boundary = 1.83 - 0.85 * front**1.5 + 0.27 * np.maximum(0, np.sin(theta))
    phi = np.arange(1, rings + 1)[:, None] / rings * boundary
    directions = np.stack(
        (np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)), axis=-1
    ).reshape(-1, 3)
    directions = np.vstack(((0.0, 0.0, 1.0), directions))
    radial = (head - center) / radius
    extent = np.linalg.norm(radial, axis=1)
    skin_normals = _unit(radial)
    lengths = np.empty(len(directions))
    for i in range(0, len(directions), 16):
        group = directions[i : i + 16]
        match = np.einsum("ij,kj->ik", group, skin_normals).argmax(axis=1)
        lengths[i : i + len(group)] = extent[match]
    out = _unit(directions / radius)
    skin = center + (lengths[:, None] * directions) * radius
    thickness = 0.045 * min(radius)
    outer = skin + out * thickness
    inner = skin - out * (0.5 * thickness)
    vertices = np.concatenate((outer, inner)) @ rotation.T + origin
    surface = []
    for j in range(sides):
        nxt = (j + 1) % sides
        surface.append((0, 1 + j, 1 + nxt))
        for k in range(rings - 1):
            r = 1 + k * sides
            surface.append((r + j, r + sides + j, r + sides + nxt, r + nxt))
    count = len(outer)
    faces = surface + [tuple(index + count for index in reversed(face)) for face in surface]
    base = count - sides
    for j in range(sides):
        a, b = base + j, base + (j + 1) % sides
        faces.append((a, a + count, b + count, b))
    return vertices, faces


class _Hair(_Particles):
    SIDES = 5

    def __init__(self, part, mesh, level, obstacles):
        rotation, v, lo, hi = _local_bounds(part, mesh)
        h = hi[2] - lo[2]
        strands, segments = ((32, 7), (58, 9), (88, 11))[level]
        head = v[v[:, 2] > hi[2] - 0.135 * h] if part.shape == "human" else v
        a, b = head.min(0), head.max(0)
        center = (a + b) / 2
        self.scalp_center = center @ rotation.T + part.center
        radius = (b - a) / 2
        radius = np.maximum(radius, 1e-5)
        grid = []
        for i in range(strands):
            az = i * 2.399963229728653
            phi = 0.17 + 0.99 * np.sqrt((i + 0.5) / strands)
            t = np.linspace(0, 1, segments)
            # Sweep the front hair towards a temple, leaving eyes and lips
            # visible. The first segment sets the direction of scalp growth.
            front = max(0.0, -np.sin(az))
            sweep = np.sign(np.cos(az)) * front * (1.32 * t**0.72)
            azimuth = az + sweep
            angle = phi + t * (1.19 + 0.21 * np.sin(az) - 0.25 * front)
            direction = np.stack(
                (
                    np.sin(angle) * np.cos(azimuth),
                    np.sin(angle) * np.sin(azimuth),
                    np.cos(angle),
                ),
                axis=1,
            )
            curve = center + radius * direction * (1 + 0.065 * np.sin(pi * t)[:, None])
            if part.shape == "human" or part.shape in ("head", "sphere"):
                root = head[np.argmin(np.sum(((head - curve[0]) / radius) ** 2, axis=1))]
            else:
                root = np.array((curve[0, 0], curve[0, 1], hi[2]))
            curve += (root - curve[0]) * (1 - t[:, None])
            curve[:, 2] -= radius[2] * 0.53 * t * t
            grid.append(curve @ rotation.T + part.center)
        grid = np.asarray(grid)
        pinned = np.zeros((strands, segments), bool)
        pinned[:, :2] = True
        self.radius = max(1e-6, min(radius[:2]) * 0.15)
        self.initialize(
            grid,
            pinned,
            ((0, 1, 1), (0, 2, 0.64)),
            obstacles,
            min(0, mesh.vertices[:, 2].min()),
            0.34 * self.radius,
        )
        # Styled hair bundles retain bending memory. Anchor the graft tangent
        # and gently restore their groomed curvature, tapering off at the tips.
        self.groom = np.tile(np.linspace(0.060, 0.010, segments), strands)[:, None]
        self.faces = []
        ring_count = segments - 1
        for strand in range(strands):
            start = strand * (ring_count * self.SIDES + 2)
            tip = start + ring_count * self.SIDES + 1
            for k in range(self.SIDES):
                nxt = (k + 1) % self.SIDES
                self.faces.append((start, start + 1 + nxt, start + 1 + k))
                for ring in range(ring_count - 1):
                    i = start + 1 + ring * self.SIDES
                    self.faces.append(
                        (i + k, i + nxt, i + self.SIDES + nxt, i + self.SIDES + k)
                    )
                i = tip - self.SIDES
                self.faces.append((i + k, i + nxt, tip))
        self.scalp = None
        if part.shape in ("head", "human", "sphere"):
            self.scalp, patch = _scalp(head, center, radius, rotation, part.center, level)
            offset = strands * (ring_count * self.SIDES + 2)
            self.faces.extend(tuple(i + offset for i in f) for f in patch)
        self.faces = tuple(self.faces)
        self.smooth = (True,) * len(self.faces)

    def mesh(self):
        grid = self.pos.reshape(*self.grid_shape, 3)
        tangent = _unit(np.gradient(grid, axis=1))
        # Flatten each ribbon along the scalp and transport its frame to the tip.
        outward = _unit(grid[:, 0] - self.scalp_center)
        aligned = abs(np.einsum("ij,ij->i", outward, tangent[:, 0])) > 0.98
        outward[aligned] = np.eye(3)[np.abs(tangent[aligned, 0]).argmin(axis=1)]
        side = np.empty_like(tangent)
        side[:, 0] = _unit(np.cross(outward, tangent[:, 0]))
        for i in range(1, grid.shape[1]):
            side[:, i] = _unit(
                side[:, i - 1]
                - (side[:, i - 1] * tangent[:, i]).sum(axis=1)[:, None] * tangent[:, i]
            )
        normal = np.cross(tangent, side)
        angle = 2 * pi * np.arange(self.SIDES) / self.SIDES
        taper = self.radius * np.linspace(1, 0.30, grid.shape[1] - 1)
        taper[0] *= 0.25
        offset = (
            side[:, :-1, None, :] * np.cos(angle)[None, None, :, None]
            + 0.32 * normal[:, :-1, None, :] * np.sin(angle)[None, None, :, None]
        ) * taper[None, :, None, None]
        centers = grid[:, :-1].copy()
        centers[:, 0] -= 0.21 * self.radius * outward
        rings = (centers[:, :, None, :] + offset).reshape(len(grid), -1, 3)
        points = np.concatenate((centers[:, :1], rings, grid[:, -1:]), axis=1).reshape(-1, 3)
        if self.scalp is not None:
            points = np.concatenate((points, self.scalp))
        return _frame(points, self.faces, self.smooth)


class _Water:
    def __init__(self, mesh, level):
        lo, hi = mesh.vertices.min(0), mesh.vertices.max(0)
        size = np.maximum(hi[:2] - lo[:2], 1e-6)
        intervals = np.maximum(3, np.rint((11, 19, 29)[level] * size / size.max()).astype(int))
        nx, ny = intervals + 1
        x, y = np.meshgrid((np.arange(nx) + 0.5) / nx, (np.arange(ny) + 0.5) / ny)
        self.xy = np.stack((lo[0] + size[0] * x, lo[1] + size[1] * y), axis=-1)
        self.dx, self.dy = size / np.array((nx, ny))
        self.depth = 0.11 * min(size)
        self.bottom = hi[2]
        self.elevation = (
            self.depth * 0.29 * np.exp(-((x - 0.31) ** 2 / 0.014 + (y - 0.58) ** 2 / 0.035))
        )
        self.elevation += self.depth * (
            0.13 * np.cos(5 * pi * x) * np.cos(2 * pi * y) + 0.065 * np.cos(3 * pi * y)
        )
        self.elevation -= self.elevation.mean()
        self.u = np.zeros((ny, nx + 1))
        self.v = np.zeros((ny + 1, nx))
        self.faces = _closed_sheet(ny, nx)
        self.smooth = (True,) * ((nx - 1) * (ny - 1)) + (False,) * (
            len(self.faces) - (nx - 1) * (ny - 1)
        )
        # Scale the wind stress by depth / basin length. Its equilibrium
        # surface tilt then stays in the linear wave regime at any scene scale.
        self.push = (
            (self.depth / size[0]) * 0.09 * np.cos(pi * x[:, :-1]) * np.sin(pi * y[:, :-1])
        )

    def snapshot(self):
        return self.elevation.copy(), self.u.copy(), self.v.copy()

    def restore(self, state):
        self.elevation[:], self.u[:], self.v[:] = state

    def step(self, frame, wind, gravity):
        c = np.sqrt(gravity * self.depth)
        substeps = max(1, ceil(c / FPS * np.hypot(1 / self.dx, 1 / self.dy) / 0.70))
        if substeps > 128:
            raise ValueError("Размер области жидкости требует слишком малого шага времени")
        dt = 1 / (FPS * substeps)
        damping = np.exp(-0.38 * dt)
        for j in range(substeps):
            self.u[:, 1:-1] *= damping
            self.v[1:-1] *= damping
            self.u[:, 1:-1] += dt * (
                -gravity * np.diff(self.elevation, axis=1) / self.dx
                + wind * (gravity / 9.81) * self.push * np.cos(2 * (frame + j / substeps) / FPS)
            )
            self.v[1:-1] -= dt * gravity * np.diff(self.elevation, axis=0) / self.dy
            self.elevation -= (
                dt
                * self.depth
                * (np.diff(self.u, axis=1) / self.dx + np.diff(self.v, axis=0) / self.dy)
            )

    def mesh(self):
        height = self.bottom + self.depth + self.elevation
        top = np.dstack((self.xy, height)).reshape(-1, 3)
        bottom = top.copy()
        bottom[:, 2] = self.bottom
        return _frame(np.concatenate((top, bottom)), self.faces, self.smooth)


class Simulation:
    def __init__(self, kind, part, mesh, detail, colliders=(), wind=1.0, strength=1.0):
        if kind not in KINDS or detail not in DETAILS:
            raise ValueError("Неизвестный тип или детализация симуляции")
        if (
            type(wind) not in (int, float)
            or not -8 <= wind <= 8
            or type(strength) not in (int, float)
            or not 0.1 <= strength <= 2
        ):
            raise ValueError("Ветер должен быть в пределах ±8 м/с², гравитация — 0,1–2 g")
        level = DETAILS.index(detail)
        self.replace_selected = kind == "cloth" and part.shape == "cloak"
        colliders = tuple((p, m) for p, m in colliders if p is not part)
        source_triangles = sum(len(f) - 2 for f in mesh.faces)
        count = sum(sum(len(f) - 2 for f in other.faces) for _, other in colliders)
        estimates = {
            "cloth": source_triangles if self.replace_selected else (500, 1050, 2050)[level],
            "water": (600, 1610, 3630)[level],
            "hair": (2320, 5480, 10000)[level],
            "fluid": TRIANGLE_LIMITS[level],
        }
        required = count + estimates[kind] + (0 if self.replace_selected else source_triangles)
        temporary = texture_working_bytes([part, *(p for p, _ in colliders)])
        if kind == "fluid":
            temporary += 8 * 1024**2
        if required > mesh_triangle_budget(temporary):
            raise ValueError(
                "Для симуляции недостаточно свободной памяти. Уменьшите детализацию."
            )
        if len(mesh.vertices) < 3 or not np.isfinite(mesh.vertices).all():
            raise ValueError("Некорректная исходная сетка")
        self.kind, self.name = kind, part.name if self.replace_selected else _NAMES[kind]
        self.wind, self.gravity = wind, 9.81 * strength
        if kind == "water":
            self._solver = _Water(mesh, level)
        else:
            # The selected support and the nearest solid parts bound the collision work.
            nearest = sorted(
                colliders,
                key=lambda pair: np.linalg.norm(np.asarray(pair[0].center) - part.center),
            )[:4]
            pieces = [] if self.replace_selected else [(part, mesh)]
            pieces.extend((p, m) for p, m in nearest if p.shape not in ("cloak", "hood"))
            obstacles = [
                o
                for p, m in pieces
                for o in _obstacles(p, m, scalp=(kind == "hair" and p is part))
            ]
            if kind == "fluid":
                self._solver = ParticleFluid(mesh, level, obstacles, FPS, self.gravity)
            else:
                self._solver = (_Cloth if kind == "cloth" else _Hair)(
                    part, mesh, level, obstacles
                )
        self.frame = 0
        self._checkpoints = {0: self._solver.snapshot()}
        self._mesh_frame = -1

    def advance(self):
        if self.frame < MAX_FRAME:
            self._solver.step(self.frame, self.wind, self.gravity)
            self.frame += 1
            if self.frame % 15 == 0 and self.frame not in self._checkpoints:
                self._checkpoints[self.frame] = self._solver.snapshot()
        return self.frame

    def reset(self):
        self._solver.restore(self._checkpoints[0])
        self.frame = 0

    def seek(self, target, limit=MAX_FRAME):
        if type(target) is not int or not 0 <= target <= MAX_FRAME:
            raise ValueError("Кадр вне диапазона симуляции")
        saved = max(i for i in self._checkpoints if i <= target)
        if self.frame > target or self.frame < saved:
            self._solver.restore(self._checkpoints[saved])
            self.frame = saved
        for _ in range(min(limit, target - self.frame)):
            self.advance()
        return self.frame

    def mesh(self):
        if self._mesh_frame != self.frame:
            self._mesh = self._solver.mesh()
            self._mesh_frame = self.frame
        return self._mesh
