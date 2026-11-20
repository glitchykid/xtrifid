[English](README.md) · [Русский](README.ru.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [简体中文](README.zh-CN.md)

# Xtrifid

Describe a prop or a person, inspect the 3D mesh in GTK and revise individual parts. Xtrifid builds static, editable geometry in metres. The interface is in Russian.

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

For more complex descriptions and text edits, the studio first tries an existing local Ollama, including the running Xtrifid Docker service on Linux. Otherwise it starts its bundled CPU engine. When the first prompt finds no weights, click **Скачать модель** to download `qwen3.5:4b` (about 3.3 GB), then submit the prompt again. The built-in engine keeps weights in `~/.ollama/models` and reuses them offline. Set `OLLAMA_MODELS` for another weights directory, `OLLAMA_URL` to choose a server, or `XTRIFID_MODEL` to choose a model that supports structured output.

## Run from source with Docker

Use Docker Engine with Compose, Git, `xauth` and a desktop with X11 or XWayland and OpenGL 3.3 (Mesa software rendering works). The first launch downloads the containers. Free-form generation also needs model weights (about 3.3 GB for Qwen). Tested on a CPU system with 16 GB of RAM.

```sh
git clone https://github.com/glitchykid/xtrifid.git
cd xtrifid
./studio
```

The launcher starts Ollama and opens the studio. Use **Скачать модель** when the studio asks for weights. Later sessions reuse the model volume. Prompts stay in the local Docker network. To select another installed or downloadable Ollama model that supports structured output, set `XTRIFID_MODEL` before running `./studio`. To stop the inference service, run `docker compose down`.

## Build and refine a model

1. Use the prompt to specify a shape, dimensions and how parts meet. For example, ask for a tabletop mirror with a round disc and an oval base. For an offline first run, enter `make a person`. Choose **Game Ready** or **Для рендера**, then select **Низкая** (low), **Средняя** (medium) or **Высокая** (high) under **Детализация**. Press **Создать модель**.
2. Drag to orbit, scroll to zoom and click to select. **Сетка** shows polygon boundaries. **К детали** (F) frames the selection, **К лицу** frames a human face, and Home fits the whole model. The arrows rotate the view.
3. Change the selected part’s dimensions, position or rotation with **Применить значения**. For a person, open **Анатомия и поза** to choose a neutral, male or female adult physique, set a pose, or resize individual regions including the ears, nose, lips and chin. Click **Применить к телу**. **Изменить деталь** applies a text edit and **Отменить действие** undoes the last change.
4. **Сохранить** writes an editable `.xtri` project. **Открыть** restores it. **Экспорт OBJ** writes the current mesh. The executable saves directly to the folder you choose. In Docker, choose `/exports` to keep the file in the host’s `./exports` directory.

## Edit a corner or rim

Select a part and expand **Контур детали**. Click a point on the section drawing or choose its number with **Вершина**. The 3D view highlights the corresponding edge, circular rim or pole. Drag the point, enter its two local coordinates in metres and click **Применить координаты**, or enter a text instruction such as “Increase the radius of this point by 5 mm and keep its height.”

For a box or extrusion, the point controls a vertical edge in the local XY plane. For a cylinder, cone, tube or revolved profile, it controls a ring or pole by radius and Z. The first edit converts the primitive into an editable extrusion or revolved profile. The other control points retain their world positions. The part centre can move as its bounds change. Undo restores the previous shape. Spheres and tori support whole-part editing. Existing XTRI projects open directly.

For a curved shell, open **Поперечные сечения**. Choose a ring, set its height, X/Y radii or centre in metres and click **Применить сечение**. The highlighted ring moves while the other rings keep their positions.

## Geometry profiles

| Profile | Mesh |
| --- | --- |
| Game Ready | Consistently wound triangles. Round sections use 8, 16 or 32 segments at low, medium or high detail. A box uses 8 vertices and 12 triangles. |
| Для рендера | Round sections use 24, 48 or 64 segments. Regular surfaces retain quad strips. Poles, rims and complex caps use triangles where needed. |

The local language model can assemble up to 48 named parts and describe curved shells with elliptical sections that shift, widen and narrow along their height. Dimensions and support relationships guide placement. Its interpretation of an unfamiliar shape may require correction.

The adult figure uses a MakeHuman anatomical mesh. The ears, lips, eyelids, fingers and torso belong to one continuous skin surface. The eyes have their own geometry. A plain person is unclothed. Request a cloak or hood explicitly. Select the body to adjust its proportions. OBJ exports static geometry with no materials or animation rig.

## Detail and available resources

Choose low, medium or high detail yourself. The human mesh has six resolutions across the two profiles. A project remembers your choice. Versions 1 and 2 remain readable. Basic figure prompts work offline. Free-form descriptions and text edits require Ollama weights.

Editable examples · [Person](https://github.com/glitchykid/xtrifid/releases/download/v0.5.0/human.xtri) · [Mouse shell](https://github.com/glitchykid/xtrifid/releases/download/v0.5.0/mouse-shell.xtri) · [Tabletop mirror](https://github.com/glitchykid/xtrifid/releases/download/v0.5.0/table-mirror.xtri)

Before preview or export, Xtrifid estimates triangle count against available RAM and Linux container limits. It reserves memory for the desktop and caps each scene at 150,000 triangles after triangulation. If the requested level exceeds the current budget, the previous scene stays available. Choose a lower level or Game Ready. When opening a saved project, you can accept a level that fits. Inference is limited separately by model size, memory, CPU threads and context length. Other applications can still consume system resources.

## Troubleshooting

- **The executable does not open.** Check execute permission, the graphical session and `ldd --version` for your glibc version. If Linux reports a missing `GLIBC` version, use the Docker instructions above.
- **The model download was interrupted.** Check the server’s free disk space and internet connection. A TLS error calls for checking the server’s trusted CA certificates. Then click **Скачать модель** again. A first CPU generation can take several minutes.
- **Docker** · **Black preview or `MESA: error: Failed to attach to x11 shm`.** In a Git checkout, run `git pull --ff-only` and restart `./studio`. For an archive installation, use version 0.2.1 or later. The launcher rebuilds the image with a Mesa driver that can display frames across the Docker IPC boundary.
- **Docker** · **No window.** Launch from the graphical session and check `echo "$DISPLAY"` and `xauth info`. The launcher forwards the current display cookie.
- **Docker** · **Ollama error or a slow first request.** Run `docker compose logs --tail 40 ollama` and `docker compose exec ollama ollama list`. A cold model takes time to load on CPU. The previous scene remains available after a failed request.
- **Unexpected geometry.** Specify quantities, units and contact points. Inspect the silhouette and mesh, then correct the selected part.

Report problems with the prompt, profile and reproduction steps. [GitHub Issues](https://github.com/glitchykid/xtrifid/issues) · [Architecture and Docker tests](docs/architecture.md)
