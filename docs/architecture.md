# Local generation and editable topology

## Decision

Xtrifid constructs static meshes in metres. Ollama translates free text into a bounded geometry recipe. An object can contain named solids with explicit dimensions, sections and support relationships. A human recipe sets physique, posture, stature and local proportions on a bundled anatomical surface. A garment is a separate part.

Curved shells can be described by elliptical cross-sections with independently varying radii and offsets. Sections share perimeter loops. The shape accepts flat or rounded ends and tips. Other solids include extruded simple polygons, tubes and surfaces of revolution. Concave caps use ear clipping. Game Ready triangulates the final mesh. Render keeps quads on regular surfaces and preserves the anatomical edge loops around the mouth, ears and eyes at higher detail. Separate parts can intersect.

## Data flow

1. The user chooses the profile and detail. GTK sends the prompt to a worker. A whole-prompt grammar handles common Russian and English figure requests offline. For other descriptions Ollama identifies a figure, an object assembled from parts, or a single shaped shell. For a composite object the semantic pass can identify named parts, their supports, relative widths and whether they meet or overlap. Each support refers to a named part, the floor or free placement. If the object classifier returns invalid component relationships twice, the geometry pass uses the original description and selected detail. Geometry validation and the scene limits apply on both paths. The semantic response budget ranges from 192 to 768 tokens within the 2,048-token context. The client checks the returned coordinates, dimensions, topology parameters and references before building any mesh. Invalid descriptions can be retried once.
2. The assembly resolver checks references and cycles, then places each supported part on the specified horizontal support plane. The geometry engine applies scaling followed by XYZ Euler rotation and translation.
3. The document owns immutable part descriptions and up to 20 undo states. AI edits return a patch for one selected part. Numeric edits preserve untouched values, including precision beyond the displayed digits. A loft section edit works in metres and refits the changed part’s bounds around its existing world-space rings.
4. The GTK OpenGL viewport and the OBJ and GLB writers consume the same meshes. The viewport triangulates quads for display and uses the original polygon boundaries for its wire overlay. Display lighting stays in the viewport. The camera remembers the selected part as its orbit centre in close-up. For a human it frames the head and follows its orientation. Rendering and picking use the same transform.
5. XTRI version 4 keeps the profile, detail, Low Poly style, per-part geometry, anatomy and colour recipe. Version 5 also stores physical surface parameters and optional rest UVs on a baked surface. The writer uses v5 when either feature is present. Versions 1–4 still load with their original surface settings. A snapshot stores normalised float64 vertices, face indices and smoothing flags. Counts, dimensions and the 16 MiB file cap are checked before construction. The version 1 reader assigns medium detail. Atomic replacement protects the previous save if writing fails.

## Surface reconstruction, physics and colour

An organic recipe is a connected graph of up to 24 named nodes with local positions and radii. Tapered segments form one implicit surface. A silhouette recipe retains up to two binary orthographic masks of at most 96 × 96 samples. A front/right pair defines a visual hull. Rounded reconstruction estimates cross-sections from both profiles. Where projections split around an opening, the envelope of inscribed balls estimates local thickness. Smoothing is constrained by the original silhouette and topology. The regular surface grid uses 12–44 cells per axis before local adjustments for ambiguous branching configurations. Surface Nets emits quads on unambiguous cells. The extractor falls back to consistently split tetrahedra for unresolved configurations. It counts triangles before allocating faces. An exact rectangular silhouette hull uses eight vertices. Game Ready triangulates at export. Low Poly lowers the sampling density and disables smooth normals. [Surface Nets construction](https://0fps.net/2012/07/12/smooth-voxel-terrain-part-2/) · [Polygonising a scalar field](https://paulbourke.net/geometry/polygonise/)

GdkPixbuf 2.42.12 in the development image decodes local PNG and JPEG files after checking file type, dimensions, byte size and available memory. The longest preview edge is at most 448 px. Transparent images use alpha. Opaque silhouettes use the median border colour and a selectable tolerance. Reference editing and reopening use the masks stored in the project. For ordinary photos, the bounded vision request sends a JPEG to the selected Ollama endpoint. With two views the request uses a single composite, preserving left and right placement. The vision model describes the object before a text pass emits geometric operations and named supports. A photo can supply up to twelve observed volumes in the initial assembly. For elongated attachments or upright handles, one optional visual pass can refine placement and geometry. A failed optional pass keeps the validated assembly. Organic forms and figures can use the ordinary text planner. Hidden details can be missing or misjudged. The studio rescales the validated scene to the requested height. [GdkPixbuf loader](https://docs.gtk.org/gdk-pixbuf/class.PixbufLoader.html) · [GdkPixbuf compositing](https://docs.gtk.org/gdk-pixbuf/method.Pixbuf.composite.html) · [Ollama vision input](https://docs.ollama.com/capabilities/vision)

Simulation advances in fixed steps of 1/30 s over six seconds. Cloth uses positional constraints, hair uses rooted strand guides, the water surface uses shallow-water waves and a falling liquid volume uses a bounded position-based particle solver. Detail limits the grid or particle count. During playback a separate OpenGL buffer holds the calculated frame and the rest of the scene keeps its buffers. A bake uses the visible vertices, polygons and winding. It can replace a selected cloak while preserving its material, or insert an independent part. The resulting static surface is undoable, editable as a part and exportable. [Position Based Dynamics](https://matthias-research.github.io/pages/publications/posBasedDyn.pdf) · [Position Based Fluids](https://mmacklin.com/pbf_sig_preprint.pdf) · [Shallow water equations](https://www.clawpack.org/riemann_book/html/Shallow_water.html)

Materials store a seed, bounded resolution, colour palette, scale and pattern. A free-text request can ask the local model to choose this procedure. The material renderer generates the same RGB pixels and per-face UV charts for GTK and Wavefront export. The OBJ writer publishes a complete content-addressed directory of PNG files and a relative MTL before replacing the OBJ. Exports that share a directory retain access to their own material sets. [PNG specification](https://www.w3.org/TR/png-3/) · [Wavefront OBJ](https://www.loc.gov/preservation/digital/formats/fdd/fdd000507.shtml)

## Game material and tangent convention

PBR materials keep three scalars alongside the colour recipe. Each substrate defines a periodic height field. Its filtered derivatives generate unit tangent-space normals and its exposed surface modulates roughness. Pigment and printed motifs affect base colour. Normal strength is baked once. Roughness and metallic are stored as their final linear values in the green and blue channels. Base colour alone is sampled as sRGB. The program uses the same generators for file output and preview. [Khronos glTF 2.0 materials](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#materials) · [PNG colour types](https://www.w3.org/TR/png-3/#11IHDR)

OpenGL 3.3 shades PBR surfaces with studio illumination, a GGX specular response and the mesh's signed tangent frame. Hard edges and mirrored UVs split the frame. Positive green points toward the top of the image. GTK and OBJ use bottom-left UVs. GLB converts the UV origin, preserves tangent handedness and rotates Z-up scene coordinates to glTF Y-up in metres. The rest chart follows simulated fabric through deformation, baking and reopening the project. [OpenGL 3.3 specification](https://registry.khronos.org/OpenGL/specs/gl/glspec33.core.pdf) · [Khronos Blender 4.3 glTF exporter](https://github.com/KhronosGroup/glTF-Blender-IO/blob/blender-v4.3-release/addons/io_scene_gltf2/blender/exp/primitive_extract.py)

GLB embeds the mesh, indices and PNGs in a single file. Equal local meshes with the same material share geometry between named nodes. The writer stages one part and one map at a time and publishes the completed file atomically. Game Ready OBJ uses the MTL extensions `norm`, `map_Pr` and `map_Pm` with separate data images. Legacy colour materials retain their diffuse output. [glTF 2.0 binary files](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#binary-gltf-layout) · [TinyObjLoader PBR MTL extension](https://github.com/tinyobjloader/tinyobjloader/blob/release/pbr-mtl.md)

The material budget counts retained textures for each unique recipe and reserves working memory for generation. A 1024² PBR tile is budgeted at 24 MiB resident and 96 MiB scratch. On a single Docker CPU the six presets at two pattern scales took 0.18–0.29 s each to rasterize their three 1024² maps. A test under a 660 MiB container limit refused 20 distinct 1024² materials before meshing and retained the previous scene. Larger textures are rasterized by one worker per viewport, with OpenGL uploads on the GTK thread. [GTK 4 GLArea](https://docs.gtk.org/gtk4/class.GLArea.html) · [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)

## Decision on figures and resource limits

The bundled MakeHuman hm08 skin connects the face, ears, fingers, limbs and torso. Separate ocular surfaces fit the sockets. The six mesh levels are prepared offline, so high detail never triggers subdivision of an unbounded mesh at runtime. Posing uses sampled limb weights. Local edits deform bounded regions of the skin. The OBJ contains the posed vertices and original polygon boundaries. Neutral, male and female adult shapes share topology; unpacking a morph adds only its displacement vectors. Authoring, attribution and pinned inputs are documented with the assets. [Anatomical assets](../xtrifid/assets/README.md)

Game Ready uses 8, 16 or 32 radial segments, and Render uses 24, 48 or 64. Before constructing a preview or writing an OBJ, the engine sums conservative triangle bounds over all parts. It combines Linux `MemAvailable` with visible cgroup v2 memory limits and current usage. The host reserve is 20% of physical RAM, clamped to 1.5–4 GiB. The mesh budget allows 8 KiB of temporary working space per triangle after a further 256 MiB reserve, up to 150,000 triangles. If capacity cannot be measured reliably, allocation is refused. A replacement preview is prepared before changing the document and undo history. A saved project that exceeds the current budget can open at a lower level with the user's confirmation. [Linux cgroup v2](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html) · [GTK alert choices](https://docs.gtk.org/gtk4/method.AlertDialog.choose.html) · [NumPy 1.26 array concatenation](https://numpy.org/doc/1.26/reference/generated/numpy.concatenate.html) · [NumPy 2.2 array concatenation](https://numpy.org/doc/2.2/reference/generated/numpy.concatenate.html) Dense transformed meshes bypass the small-part cache so repeated edits release the previous vertex and face arrays.

The inference client inspects installed weights and running models before submitting chat. It reserves space for the runner and context, reduces context and thread counts to fit, and admits one inference per studio process. The Docker launcher calculates a service ceiling of up to 6 GiB and four CPUs, disables service swap and permits one active model. In native mode the client monitors the bundled worker's process group. During generation the client checks available memory and can interrupt the request. Controls on external Ollama services depend on their administrator and resource measurements can change as other applications allocate memory. [Ollama v0.40.16 API types](https://github.com/ollama/ollama/blob/v0.40.16/api/types.go) · [List installed models](https://docs.ollama.com/api/tags) · [List running models](https://docs.ollama.com/api/ps) · [Docker Compose service limits](https://docs.docker.com/reference/compose-file/services/)

## Local profile edits

The profile editor shares the same cross section and rotation matrix used by the mesh generator. A control point identifies a vertical edge of an extruded polygon, or a ring or pole of a turned surface. The highlighted polyline comes from the actual mesh for the active resolution. The canvas keeps drag previews local and commits one undoable edit on release.

For a text edit, Ollama identifies the axes, the unit (mm, cm or m) and whether the requested coordinate is absolute or relative. The client converts units and computes the new coordinates. Axes omitted by the user retain their exact values.

Edits use local metres. After a change, the geometry module validates the contour, recomputes its dimensions and rotates the shift of the bounding-box centre into world space. This keeps other control points anchored, including on rotated and elliptical parts. The first edit of a primitive uses the extrusion or lathe representation introduced in XTRI version 1. A no-op preserves the original primitive. Self-intersections, inverted winding, invalid axis contact and out-of-range dimensions are rejected.

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

The independent `tests/native_gui.py` driver opens a version 1 project by filename, samples the actual X11 framebuffer, switches mesh profile, chooses low detail and generates a cloaked person. It inspects the exported OBJ topology and measures the figure's height. The same driver imports an actual PNG through the native file picker, builds a Low Poly ring, verifies the hole after OBJ export, checks the diffuse PNG, imports the exported GLB structure and its PBR assignments, then bakes a cloth frame. Its docstring lists the external dependencies and `--generate` option.

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

The following commands exercise the complete GTK cloak workflow and a separate, memory-constrained desktop. The second session has two CPUs, a 660 MiB memory limit and no swap or network. It generates a low-detail Game Ready mesh and requests high detail, then verifies that the lower mesh is still usable and can be exported.

```sh
docker compose run --rm --no-deps -e PYTHONPATH=/app tests xvfb-run -a dbus-run-session -- python3 tests/live_character.py
docker run --rm --init --network none --memory=660m --memory-swap=660m --cpus=2 -e LP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=1 -e PYTHONPATH=/app xtrifid-tests xvfb-run -a dbus-run-session -- python3 tests/live_character.py --constrained
```

With the Qwen weights present, run `docker compose run --rm --no-deps -e PYTHONPATH=/app tests python3 tests/live_character_planner.py` to exercise free-form character and prop descriptions through the live model.

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

For live examples of a curved mouse enclosure and a three-piece tabletop mirror on the installed language model, run the following in Docker. The checks measure the exported geometry, including the crest and the disc outline.

```sh
docker compose run --rm --no-deps -v ./exports:/exports -e PYTHONPATH=/app tests python3 tests/live_freeform.py /exports mouse-shell
docker compose run --rm --no-deps -v ./exports:/exports -e PYTHONPATH=/app tests python3 tests/live_freeform.py /exports table-mirror
```

Branch and image generation can also run against the installed model. The coral check measures three connected branches from a Russian prompt. The photo check accepts a local PNG or JPEG, with an optional second view via `--right`. For the published espresso photograph, `--check-espresso` checks the handle opening, vessel and saucer in the geometry. Place the photograph in `exports/coffee.png`. [scikit-image coffee photograph](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.coffee)

```sh
docker compose run --rm --no-deps -v ./exports:/exports -e PYTHONPATH=/app tests python3 tests/live_sculpt.py /exports/coral.xtri
docker compose run --rm --no-deps -v ./exports:/exports -e PYTHONPATH=/app tests python3 tests/live_photo.py /exports/coffee.png --check-espresso --output /exports/coffee.xtri
```
