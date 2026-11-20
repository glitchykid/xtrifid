"""Small free-surface Position Based Fluids solver (Macklin & Müller, 2013).

Positions are world metres. The density kernel is evaluated in particle spacings
so its rest density and regularisation do not change with the size of a scene.
"""

from math import ceil

import numpy as np

from .geometry import Mesh
from .sculpt import isosurface, isosurface_triangle_count

# The surface coarsens before triangulation to stay within this reservation.
TRIANGLE_LIMITS = (4600, 7500, 11000)


class ParticleFluid:
    SMOOTH_RADIUS = 1.82

    def __init__(self, mesh, level, obstacles, fps, gravity):
        a, b = mesh.vertices.min(0), mesh.vertices.max(0)
        width = min((b - a)[:2])
        self.fps = fps
        self.obstacles = obstacles
        nx, ny, nz = ((5, 5, 5), (6, 6, 6), (7, 7, 7))[level]
        self.spacing = 0.34 * width / max(nx, ny)
        if self.spacing <= 0:
            raise ValueError("Для жидкости нужна поверхность ненулевой ширины")
        self.substeps = max(3, ceil(np.sqrt(gravity / self.spacing) / (self.fps * 0.22)))
        if self.substeps > 14:
            raise ValueError(
                "Для такого малого объёма требуется слишком малый шаг. Увеличьте выбранную поверхность."
            )
        drop = np.indices((nx, ny, nz)).reshape(3, -1).T.astype(float)
        drop -= drop.mean(0)
        drop = drop[np.linalg.norm(drop / np.array((nx, ny, nz)), axis=1) < 0.51]
        phase = np.arange(len(drop))
        # A slightly tilted, staggered pour avoids an exactly aligned lattice
        # landing all its lower particles simultaneously.
        drop[:, 0] += 0.24 * drop[:, 2] + 0.11 * np.cos(phase * 3.3)
        drop[:, 1] += 0.09 * np.sin(phase * 5.4)
        drop[:, 2] += 0.13 * drop[:, 0]
        center = (a + b) / 2
        center[2] = b[2] + 0.56 * width
        self.pos = center + drop * self.spacing
        self.vel = np.zeros_like(self.pos)
        self.vel[:, 0] = 0.09 * np.sqrt(9.81 * width)
        cx, cy = center[:2]
        self.bounds = np.array(
            (
                (cx - 0.85 * width, cy - 0.85 * width, max(min(0.0, a[2]), b[2] - 0.4 * width)),
                (cx + 0.85 * width, cy + 0.85 * width, b[2] + 1.45 * width),
            )
        )
        self.left, self.right = np.triu_indices(len(drop), 1)
        self.mass = np.ones(len(drop))
        neighbor_grid = np.indices((5, 5, 5)).reshape(3, -1).T - 2
        self.density0 = np.sum(
            np.maximum(0, 1.0 - (neighbor_grid**2).sum(1) / self.SMOOTH_RADIUS**2) ** 3
        )
        self.triangle_limit = TRIANGLE_LIMITS[level]
        self.wave_radius = self.spacing * 1.75
        self.surface_step = self.spacing * (0.87, 0.76, 0.68)[level]

    def snapshot(self):
        return self.pos.copy(), self.vel.copy()

    def restore(self, state):
        self.pos[:], self.vel[:] = state

    def _sum(self, ids, values):
        n = len(self.pos)
        return np.column_stack(
            [np.bincount(ids, weights=component, minlength=n) for component in values.T]
        )

    def _density_projection(self, left, right):
        delta = (self.pos[left] - self.pos[right]) / self.spacing
        q = 1.0 - np.einsum("ij,ij->i", delta, delta) / self.SMOOTH_RADIUS**2
        within = q > 0
        i, j = left[within], right[within]
        q = q[within]
        kernel = q**3
        n = len(self.pos)
        density = (
            self.mass
            + np.bincount(i, weights=kernel, minlength=n)
            + np.bincount(j, weights=kernel, minlength=n)
        )
        gradient = (
            (-6 / (self.SMOOTH_RADIUS**2 * self.density0)) * delta[within] * (q * q)[:, None]
        )
        net = self._sum(i, gradient) - self._sum(j, gradient)
        norm2 = np.einsum("ij,ij->i", gradient, gradient)
        denom = (
            np.einsum("ij,ij->i", net, net)
            + np.bincount(i, weights=norm2, minlength=n)
            + np.bincount(j, weights=norm2, minlength=n)
        )
        pressure = -np.maximum(0, density / self.density0 - 1) / (denom + 0.018)
        correction = (pressure[i] + pressure[j])[:, None] * gradient
        self.pos += self.spacing * (self._sum(i, correction) - self._sum(j, correction))
        return i, j, kernel

    def step(self, frame, wind, gravity):
        substeps = self.substeps
        dt = 1 / (self.fps * substeps)
        clearance = 0.52 * self.spacing
        for sub in range(substeps):
            old = self.pos.copy()
            self.vel += dt * np.array(
                (
                    wind * 0.12,
                    wind * 0.025 * np.sin((frame + sub / substeps) / self.fps * 3),
                    -gravity,
                )
            )
            self.pos += dt * self.vel
            # Keep only near-neighbour pairs in the iterative density solve.
            delta = self.pos[self.left] - self.pos[self.right]
            near = (
                np.einsum("ij,ij->i", delta, delta)
                < (self.spacing * (self.SMOOTH_RADIUS + 0.8)) ** 2
            )
            left, right = self.left[near], self.right[near]
            for _ in range(3):
                i, j, w = self._density_projection(left, right)
                for obstacle in self.obstacles:
                    obstacle.project(self.pos, clearance)
                np.clip(
                    self.pos,
                    self.bounds[0] + clearance,
                    self.bounds[1] - clearance,
                    out=self.pos,
                )
            velocity = (self.pos - old) / dt
            exchange = (velocity[j] - velocity[i]) * w[:, None]
            self.vel = velocity + (self._sum(i, exchange) - self._sum(j, exchange)) * (
                0.24 / self.density0
            )
            self.vel *= 0.998
            # Guard the fixed search radius and keep fast individual droplets
            # from tunnelling through a support between two contact projections.
            speed = np.linalg.norm(self.vel, axis=1)
            self.vel *= np.minimum(1, 0.82 * self.spacing / (dt * np.maximum(speed, 1e-12)))[
                :, None
            ]

    def _field(self, step):
        # Compact density splats, with a zero-density border on every side.
        # The evaluation grid follows the occupied region, including free drops.
        origin = self.pos.min(0) - self.wave_radius - 2 * step
        shape = np.ceil((self.pos.max(0) + self.wave_radius - origin) / step).astype(int) + 3
        field = np.zeros(shape, dtype=np.float32)
        q = (self.pos - origin) / step
        base = np.floor(q).astype(int)
        reach = ceil(self.wave_radius / step)
        offsets = np.indices((2 * reach + 1,) * 3).reshape(3, -1).T - reach
        indices = base[:, None, :] + offsets
        delta = (indices - q[:, None, :]) * (step / self.wave_radius)
        d2 = np.einsum("ijk,ijk->ij", delta, delta)
        mask = d2 < 1
        np.add.at(field, tuple(indices[mask].T), (1.0 - d2[mask]) ** 3)
        field -= 0.52
        # Clip the reconstructed free surface to solid obstacles. Particle
        # centres alone cannot keep the finite kernel from entering a support.
        occupied = tuple(np.nonzero(field > -0.52))
        locations = np.column_stack(occupied) * step + origin
        clearance = locations[:, 2] - self.bounds[0, 2]
        for obstacle in self.obstacles:
            np.minimum(clearance, obstacle.clearance(locations), out=clearance)
        field[occupied] = np.minimum(field[occupied], clearance / self.spacing)
        return field, origin

    def mesh(self):
        step = self.surface_step
        for _ in range(5):
            field, origin = self._field(step)
            if isosurface_triangle_count(field) <= self.triangle_limit:
                vertices, faces = isosurface(field, max_triangles=self.triangle_limit)
                if not faces:
                    raise ValueError("Детализация слишком мала для поверхности жидкости")
                vertices = vertices * step + origin
                vertices.setflags(write=False)
                return Mesh(vertices, tuple(faces), (True,) * len(faces))
            step *= 1.22
        raise ValueError("Поверхность жидкости превысила доступный бюджет сетки")
