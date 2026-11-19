# Local generation and editable topology

## Decision

Xtrifid targets static props assembled from identifiable parts. The language model emits a bounded JSON description containing named solids, dimensions, transforms and support relationships. The client validates each response and resolves the support graph against geometric bounds. A generated project stores the resulting transforms. Later edits can therefore move a single part independently. This design trades sculptural freedom for predictable dimensions, closed component meshes and stable part selection.

The allowed primitives include tubes with a dimensioned bore, extruded simple polygons and surfaces of revolution. Concave caps use ear clipping, curved walls use quad strips and poles use single vertices. Game Ready triangulates the final faces and uses a lower radial resolution. Render preserves quads on regular surfaces. Each object is independent. Assembly-wide Boolean union, intersection cleanup and deformation topology are outside the current scope.

## Data flow

1. GTK sends a text request to a background worker. Ollama returns data constrained by a JSON schema. The planner limits input, output and wait time. It retries a malformed geometric description once.
2. The assembly resolver checks references and cycles, then places each supported part on the specified horizontal support plane. The geometry engine applies scaling followed by XYZ Euler rotation and translation.
3. The document owns immutable part descriptions and up to 20 undo states. AI edits return a patch for one selected part. Numeric edits preserve untouched values, including precision beyond the displayed digits.
4. The GTK OpenGL viewport and OBJ writer consume the same meshes. The viewport triangulates quads for display and uses the original polygon boundaries for its wire overlay. Display lighting stays in the viewport.
5. The versioned JSON project restores editable geometry and mode. Temporary files and atomic replacement protect a saved file if writing fails.

## Runtime

| Component | Verified target |
| --- | --- |
| Distribution | Debian 13 trixie, pinned base image |
| GUI and preview | GTK 4.18.6, PyGObject 3.50.0, OpenGL 3.3 core, PyOpenGL 3.1.9 |
| Geometry and storage | Python 3.13.5 and NumPy 2.2.4 |
| Inference | Ollama 0.40.16, Qwen 3.5 4B (`d8b0f5e9760c` in the tested runtime) |
| Checks | Ruff 0.17.6, unittest, Xvfb and Mesa |

The Dockerfile pins the base and lint image digests. Compose pins the Ollama image digest. System libraries follow Debian stable updates. The launcher reuses a locally installed model tag. Verify `docker compose exec ollama ollama list` when comparing inference across machines.

API references for the target families and upstream versions are available in the official documentation. [GTK 4.18.6 source](https://gitlab.gnome.org/GNOME/gtk/-/tree/4.18.6/gtk) · [GtkGLArea](https://docs.gtk.org/gtk4/class.GLArea.html) · [OpenGL 3.3](https://registry.khronos.org/OpenGL/specs/gl/glspec33.core.pdf) · [NumPy 2.2](https://numpy.org/doc/2.2/) · [Python 3.13](https://docs.python.org/3.13/) · [Ollama 0.40.16](https://github.com/ollama/ollama/releases/tag/v0.40.16) · [Structured outputs](https://docs.ollama.com/capabilities/structured-outputs) · [Qwen model](https://ollama.com/library/qwen3.5:4b)

## Verify inside Docker

```sh
docker compose config --quiet
docker compose build studio tests
docker compose run --rm --no-deps tests
```

The deterministic suite checks manifold edges and winding, nonzero triangle areas, concave and hollow shapes, bounds, contact placement, undo, file round trips, error handling and GTK picking on a virtual display.

For an end-to-end CPU check, start Ollama and load the weights, then run the real GTK controls against the service. This test checks the dimensions and support contacts of a generated stool, sends a targeted edit, compares its four legs, and exports both profiles. Model responses depend on the selected weights.

```sh
docker compose up -d --wait ollama
docker compose exec -T ollama ollama pull qwen3.5:4b
docker compose run --rm --no-deps -e PYTHONPATH=/app tests xvfb-run -a dbus-run-session -- python3 tests/live_studio.py
```
