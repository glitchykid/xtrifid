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

## Local profile edits

The profile editor shares the same cross section and rotation matrix used by the mesh generator. A control point identifies a vertical edge of an extruded polygon, or a ring or pole of a turned surface. The highlighted polyline comes from the actual mesh for the active resolution. The canvas keeps drag previews local and commits one undoable edit on release.

For a text edit, Ollama identifies the axes, the unit (mm, cm or m) and whether the requested coordinate is absolute or relative. The client converts units and computes the new coordinates. Axes omitted by the user retain their exact values.

Edits use local metres. After a change, the geometry module validates the contour, recomputes its dimensions and rotates the shift of the bounding-box centre into world space. This keeps other control points anchored, including on rotated and elliptical parts. The first edit of a primitive uses the existing extrusion or lathe representation in the version 1 project format. A no-op preserves the original primitive. Self-intersections, inverted winding, invalid axis contact and out-of-range dimensions are rejected.

## Native Linux executable

The release build uses a single x86-64 ELF so a desktop user can launch the studio or open a project without installing the development environment. The supported baseline is Ubuntu 24.04 or Linux Mint 22 with glibc 2.39, a graphical session and OpenGL 3.3. The executable bundles Python 3.12, GTK 4.14.5, NumPy 1.26.4 and PyOpenGL 3.1.7. Graphics drivers and the desktop's trusted CA certificates come from the operating system. PyInstaller extracts the runtime into a temporary directory at startup. Each file argument opens a project in its own window through GApplication. [PyInstaller 6.22.3](https://pyinstaller.org/en/v6.22.3/operating-mode.html) · [GApplication open signal](https://docs.gtk.org/gio/signal.Application.open.html)

A configured `OLLAMA_URL` selects one server. With the default settings, the native client checks loopback, then the existing project's private Linux Docker bridge. It starts its bundled CPU Ollama and inference runner on loopback when neither service is available. It can change servers only before sending the chat request. An existing service owns its own model store. The embedded server uses `~/.ollama/models`, supports `OLLAMA_MODELS` and stops with the application. The user explicitly requests a download after a missing-model response. The UI streams progress and requires the server's final success message before offering to retry generation. [Ollama 0.40.16](https://github.com/ollama/ollama/releases/tag/v0.40.16) · [Ollama pull API](https://docs.ollama.com/api/pull)

Build the binary in Docker and check it on an offline Ubuntu desktop fixture from the repository root.

```sh
sh packaging/build.sh
docker build --platform linux/amd64 --target smoke -f packaging/Dockerfile -t xtrifid-native-smoke .
docker run --rm --init --network none -v "$PWD/exports/release/xtrifid-linux-x86_64:/artifact/xtrifid-linux-x86_64:ro" xtrifid-native-smoke
```

The companion `xtrifid-linux-x86_64-notices.tar.gz` includes third-party notices and a package inventory. The Ubuntu toolchain uses a signed, dated archive, and the Python build dependencies use verified wheel hashes. Ollama comes from the same pinned image as the Compose deployment. [Ubuntu snapshot service](https://snapshot.ubuntu.com/)

With `qwen3.5:4b` already present in the project's Docker volume, run the packaged GUI offline against the embedded CPU server. The test opens a private X11 session, enters a Russian prompt, exports the resulting block, measures its three dimensions and checks that the inference processes exit with the GUI. An optional second argument selects a different populated weights volume.

```sh
sh tests/native_offline.sh exports/release/xtrifid-linux-x86_64
```

The independent `tests/native_gui.py` driver also opens a version 1 project by filename, samples the actual X11 framebuffer, switches mesh profile and inspects OBJ topology. Its docstring lists the external driver's dependencies and `--generate` option.

## Docker runtime

| Component | Verified target |
| --- | --- |
| Distribution | Debian 13 trixie, pinned base image |
| GUI and preview | GTK 4.18.6, PyGObject and its Cairo integration 3.50.0, OpenGL 3.3 core, PyOpenGL 3.1.9 |
| Mesa EGL, GLX and DRI | Debian trixie-backports 26.1.6-1~bpo13+1 on amd64 |
| Geometry and storage | Python 3.13.5 and NumPy 2.2.4 |
| Inference | Ollama 0.40.16, Qwen 3.5 4B (`d8b0f5e9760c` in the tested runtime) |
| Checks | Ruff 0.17.6, unittest, Xvfb and Mesa |

The Dockerfile pins the base and lint image digests. Compose pins the Ollama image digest. Most system libraries follow Debian stable updates. The launcher reuses a locally installed model tag. Verify `docker compose exec ollama ollama list` when comparing inference across machines.

The X server and studio run in separate IPC namespaces. With Mesa 25.0.7 an X server advertising MIT-SHM could receive a reference to an unreachable shared-memory segment, leaving the visible window black even after a successful OpenGL draw. Mesa 26.1 checks that the segment can actually be attached and falls back to ordinary X11 image transfer. Only the Mesa packages and their matching dependencies come from the dated Debian Backports snapshot. APT verifies Debian's signatures and package checksums. The metadata expiry exception is scoped to that immutable archive. [Mesa 26.1 changes](https://docs.mesa3d.org/relnotes/26.1.0.html) · [Debian Snapshot](https://snapshot.debian.org/)

API references for the target families and upstream versions are available in the official documentation. [GTK 4.18.6 source](https://gitlab.gnome.org/GNOME/gtk/-/tree/4.18.6/gtk) · [GtkGLArea](https://docs.gtk.org/gtk4/class.GLArea.html) · [OpenGL 3.3](https://registry.khronos.org/OpenGL/specs/gl/glspec33.core.pdf) · [NumPy 2.2](https://numpy.org/doc/2.2/) · [Python 3.13](https://docs.python.org/3.13/) · [Ollama 0.40.16](https://github.com/ollama/ollama/releases/tag/v0.40.16) · [Structured outputs](https://docs.ollama.com/capabilities/structured-outputs) · [Qwen model](https://ollama.com/library/qwen3.5:4b)

## Verify inside Docker

```sh
docker compose config --quiet
docker compose build studio tests
docker compose run --rm --no-deps tests
```

The deterministic suite checks manifold edges and winding, nonzero triangle areas, concave and hollow shapes, bounds, contact placement, undo, file round trips, error handling and GTK picking on a virtual display.

Run `sh tests/external_x11.sh` after building both images to check desktop presentation. It starts Xvfb and the studio in separate containers with private IPC, samples the model's colour on the X server's framebuffer and checks for the Mesa attach error. The test uses its own X11 socket and cookie.

For an end-to-end CPU check, start Ollama and load the weights, then run the real GTK controls against the service. This test checks the dimensions and support contacts of a generated stool, sends a targeted edit, compares its four legs, and exports both profiles. Model responses depend on the selected weights.

```sh
docker compose up -d --wait ollama
docker compose exec -T ollama ollama pull qwen3.5:4b
docker compose run --rm --no-deps -e PYTHONPATH=/app tests xvfb-run -a dbus-run-session -- python3 tests/live_studio.py
```

To verify a point edit independently, load the same Ollama service and run the following command. It opens a version 1 tube in GTK, selects the upper inner ring, asks Qwen to increase its radius by 5 mm, measures all four rings of the exported meshes in both modes, reloads the project and checks undo.

```sh
docker compose run --rm --no-deps -e PYTHONPATH=/app tests xvfb-run -a dbus-run-session -- python3 tests/live_profile.py
```
