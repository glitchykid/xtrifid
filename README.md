[English](README.md) · [Русский](README.ru.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [简体中文](README.zh-CN.md)

# Xtrifid

Describe a 3D object, inspect the mesh in GTK and revise one part at a time. A local language model plans the assembly. A geometry engine builds the editable parts in metres. The interface is in Russian.

## Run on Linux

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
4. **Сохранить** writes an editable `.xtri` project. **Открыть** restores it. **Экспорт OBJ** writes the current mesh. Choose `/exports` in the file dialog to keep files in the host’s `./exports` directory.

## Geometry profiles

| Profile | Mesh |
| --- | --- |
| Game Ready | Triangles with consistent winding. Curved sections have 16 radial segments. A box uses 8 vertices and 12 triangles. |
| Для рендера | Curved sections have 48 segments. The mesh preserves quad strips and flat box faces. Poles and complex end caps use triangles where needed. |

The modeller builds up to 48 separately named solids using boxes, ellipsoids, cylinders, tubes with a measurable bore, cones, tori, extruded polygons and revolved profiles. A revolved wall profile can describe a hollow vessel. Parts can touch or intersect, and each keeps its own mesh. The current scope is static objects and hard-surface assemblies. Production character retopology, UVs, textures, materials and rigging are outside this version.

## Troubleshooting

- **No window.** Launch from the graphical session and check `echo "$DISPLAY"` and `xauth info`. The launcher forwards the current display cookie.
- **Ollama error or a slow first request.** Run `docker compose logs --tail 40 ollama` and `docker compose exec ollama ollama list`. A cold model takes time to load on CPU. The previous scene remains available after a failed request.
- **Unexpected geometry.** Specify quantities, units and contact points. Inspect the silhouette and mesh, then correct the selected part.

Report problems with the prompt, profile and reproduction steps. [GitHub Issues](https://github.com/glitchykid/xtrifid/issues) · [Architecture and Docker tests](docs/architecture.md)
