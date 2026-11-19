[English](README.md) · [Русский](README.ru.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [简体中文](README.zh-CN.md)

# Xtrifid

Describe a 3D object, inspect the mesh in GTK and revise one part at a time. A local language model plans the assembly. A geometry engine builds the editable parts in metres. The interface is in Russian.

## Run the Linux executable

Download the executable for x86-64 Linux with glibc 2.39 or newer, a graphical desktop and OpenGL 3.3. Linux Mint 22 and Ubuntu 24.04 meet the distribution requirement. Python, GTK and the CPU version of Ollama are included. [Download Xtrifid for Linux](https://github.com/glitchykid/xtrifid/releases/latest/download/xtrifid-linux-x86_64)

Run these commands in the download folder. After granting execute permission you can also double-click the file.

```sh
chmod +x ./xtrifid-linux-x86_64
./xtrifid-linux-x86_64
```

To open a saved project directly, pass its path as an argument. Each project opens in its own window. Preview, numeric edits and OBJ export work offline.

```sh
./xtrifid-linux-x86_64 "model.xtri"
```

For text generation the studio first tries an existing local Ollama, including the running Xtrifid Docker service on Linux. Otherwise it starts its bundled CPU engine. When the first prompt finds no weights, click **Скачать модель** to download `qwen3.5:4b` (about 3.3 GB), then submit the prompt again. The built-in engine keeps weights in `~/.ollama/models` and reuses them offline. Set `OLLAMA_MODELS` for another weights directory, `OLLAMA_URL` to choose a server, or `XTRIFID_MODEL` to choose a model that supports structured output.

## Run from source with Docker

Use Docker Engine with Compose, Git, `xauth` and a desktop with X11 or XWayland and OpenGL 3.3 (Mesa software rendering works). The first launch downloads the containers and around 3.3 GB of Qwen weights. Tested on a CPU system with 16 GB of RAM.

```sh
git clone https://github.com/glitchykid/xtrifid.git
cd xtrifid
./studio
```

The launcher starts Ollama, downloads `qwen3.5:4b` once and opens the studio. Later sessions reuse the model volume. Prompts stay in the local Docker network. To select another installed or downloadable Ollama model that supports structured output, set `XTRIFID_MODEL` before running `./studio`. To stop the inference service, run `docker compose down`.

## Build and refine a model

1. Under **Описание модели**, try “Create a stool with a 40 × 40 cm seat, 4 cm thick, on four 40 cm legs.” Select a profile and press **Создать модель**.
2. Drag to orbit and scroll to zoom. Click a visible part or select it under **Детали**. **Сетка** shows polygon boundaries. With the viewport focused, use the arrows to orbit, +/− to zoom and Home to fit.
3. Enter a local change such as “Change only the seat thickness to 8 cm”, then press **Изменить деталь**. For an exact transform, enter XYZ dimensions, centre or rotation and press **Применить значения**. **Отменить действие** restores the previous state.
4. **Сохранить** writes an editable `.xtri` project. **Открыть** restores it. **Экспорт OBJ** writes the current mesh. The executable saves directly to the folder you choose. In Docker, choose `/exports` to keep the file in the host’s `./exports` directory.

## Edit a corner or rim

Select a part and expand **Контур детали**. Click a point on the section drawing or choose its number with **Вершина**. The 3D view highlights the corresponding edge, circular rim or pole. Drag the point, enter its two local coordinates in metres and click **Применить координаты**, or enter a text instruction such as “Increase the radius of this point by 5 mm and keep its height.”

For a box or extrusion, the point controls a vertical edge in the local XY plane. For a cylinder, cone, tube or revolved profile, it controls a ring or pole by radius and Z. The first edit converts the primitive into an editable extrusion or revolved profile. The other control points retain their world positions. The part centre can move as its bounds change. Undo restores the previous shape. Spheres and tori support whole-part editing. Existing XTRI projects open directly.

## Geometry profiles

| Profile | Mesh |
| --- | --- |
| Game Ready | Triangles with consistent winding. Curved sections have 16 radial segments. A box uses 8 vertices and 12 triangles. |
| Для рендера | Curved sections have 48 segments. The mesh preserves quad strips and flat box faces. Poles and complex end caps use triangles where needed. |

The modeller builds up to 48 separately named solids using boxes, ellipsoids, cylinders, tubes with a measurable bore, cones, tori, extruded polygons and revolved profiles. A revolved wall profile can describe a hollow vessel. Parts can touch or intersect, and each keeps its own mesh. The current scope is static objects and hard-surface assemblies. Production character retopology, UVs, textures, materials and rigging are outside this version.

## Troubleshooting

- **The executable does not open.** Check execute permission, the graphical session and `ldd --version` for your glibc version. If Linux reports a missing `GLIBC` version, use the Docker instructions above.
- **The model download was interrupted.** Check the server’s free disk space and internet connection. A TLS error calls for checking the server’s trusted CA certificates. Then click **Скачать модель** again. A first CPU generation can take several minutes.
- **Docker** · **Black preview or `MESA: error: Failed to attach to x11 shm`.** In a Git checkout, run `git pull --ff-only` and restart `./studio`. For an archive installation, use version 0.2.1 or later. The launcher rebuilds the image with a Mesa driver that can display frames across the Docker IPC boundary.
- **Docker** · **No window.** Launch from the graphical session and check `echo "$DISPLAY"` and `xauth info`. The launcher forwards the current display cookie.
- **Docker** · **Ollama error or a slow first request.** Run `docker compose logs --tail 40 ollama` and `docker compose exec ollama ollama list`. A cold model takes time to load on CPU. The previous scene remains available after a failed request.
- **Unexpected geometry.** Specify quantities, units and contact points. Inspect the silhouette and mesh, then correct the selected part.

Report problems with the prompt, profile and reproduction steps. [GitHub Issues](https://github.com/glitchykid/xtrifid/issues) · [Architecture and Docker tests](docs/architecture.md)
