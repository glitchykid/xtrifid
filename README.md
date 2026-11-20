[English](README.md) · [Русский](README.ru.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [简体中文](README.zh-CN.md)

# Xtrifid

Build editable 3D models from text or image references in GTK. Inspect the geometry, simulate cloth, hair and liquids, and assign materials and surface detail to individual parts. Dimensions are in metres. The interface is in Russian.

Seven editable Game Ready studies for Xtrifid 0.7.0 include a moon sword, a retro blaster, a peacock fan, a marsh creature, a lantern, an orbital spell and a breaking wave. Open any `.xtri` to edit its parts or materials. [View the models](https://github.com/glitchykid/xtrifid/blob/main/examples/art-studies/gallery.jpg) · [Download XTRI and GLB files](https://github.com/glitchykid/xtrifid/releases/download/art-studies-1/xtrifid-art-studies.zip)

## Run the Linux executable

Download the executable for x86-64 Linux with glibc 2.39 or newer, a graphical desktop and OpenGL 3.3. Linux Mint 22 and Ubuntu 24.04 meet the distribution requirement. Python, GTK and the CPU version of Ollama are included. [Download Xtrifid for Linux](https://github.com/glitchykid/xtrifid/releases/latest/download/xtrifid-linux-x86_64)

Run these commands in the download folder. After granting execute permission you can also double-click the file.

```sh
chmod +x ./xtrifid-linux-x86_64
./xtrifid-linux-x86_64
```

To open a saved project directly, pass its path as an argument. Each project opens in its own window. Preview, numeric edits and OBJ or GLB export work offline.

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

## Shape and surface

**Low Poly** selects a faceted style independently of the topology profile and detail level. Curved primitives use fewer segments. Smooth shading returns when you clear the option. Planar faces and baked frames retain their topology. The segment counts in the profile table apply to the standard style.

For a branching organic form, describe a trunk, limbs, proportions and directions, for example a coral with three tapering branches. In **Узлы формы**, select a named node and edit its local position or radius. Connected branches share a closed surface.

## Simulation

Select the supporting object and expand **Физика**. Choose cloth, hair, water surface or volumetric liquid. Cloth drapes over the selected object, and a selected cloak keeps its shoulder attachment. Hair uses anchored strands. The water surface models waves in a rectangular basin. Volumetric liquid drops onto the object and spreads under gravity.

Set wind and gravity, press **Воспроизвести**, pause or scrub the six-second timeline (180 steps). **Запечь кадр в модель** keeps the visible geometry in the project for OBJ export. A cloak is replaced by its deformed mesh. Other effects add a part. Undo restores the scene. Collisions use simplified shapes. Playback speed depends on detail and available CPU. Save a baked frame to keep it after closing the project.

## Game Ready materials

Select a part and open **Текстура детали**. Choose wood, stone, fabric, skin, metal or paint. Set the palette, motif, scale, seed and resolution (128–1024 px). **Назначить** generates the maps offline. **Новый вариант** changes the seed. **Создать по описанию (ИИ)** asks local Ollama to interpret a finish such as “polished brass with fine scratches”. The project saves the editable recipe.

**PBR-материал** is enabled for a new material in Game Ready. Set **Шероховатость** (roughness, 0–1), **Металличность** (metallic, 0–1) and **Сила рельефа** (normal strength, 0–2), then apply. At zero strength the surface has a flat normal map. The preview lights the fibres, pores or grain generated by the chosen material. Silhouette and triangle count stay the same. A previously saved colour texture keeps its original appearance until you enable PBR.

Use **Экспорт GLB** in Game Ready for a single file with indexed triangles, UVs, tangents and embedded textures. Base colour is sRGB. The tangent-space normal map uses OpenGL +Y. Roughness (green) and metallic (blue) share a linear texture. For OBJ, copy the adjacent `xtrifid-materials-…` directory with the mesh. Game Ready writes separate normal, roughness and metallic PNGs for PBR parts. If the importer only reads diffuse MTL, assign the three data maps as non-colour textures. Render OBJ preserves the colour map and quad-based surfaces.

[Three-material GLB](https://github.com/glitchykid/xtrifid/releases/download/v0.7.0/material-study.glb) · [Editable project](https://github.com/glitchykid/xtrifid/releases/download/v0.7.0/material-study.xtri)

## Models from reference images

Choose a detail level and click **По референсам…**. In **Силуэты**, select a front PNG or JPEG, optionally a right view of the same object. Use transparent or uniform backgrounds and straight-on views. The white preview shows the extracted mask. Set the height in metres. A second view sets the depth from its proportions. With one view, the default depth is 55% of the width and can be overridden. For curved subjects, **Округлить сечения** estimates rounded cross-sections and the thickness of thin features. **Силуэты модели** lets you add or erase pixels, then rebuild the volume with **Применить силуэты**.

For ordinary photos choose **Фото с ИИ**. The selected Ollama vision model receives reduced images and builds an approximate editable scene from the visible subject. An optional text hint identifies the subject. Hidden surfaces are inferred. Silhouette reconstruction works offline without weights. Each image may be up to 16 MiB and 24 megapixels. In Docker, put images in the host's `exports/` and select them under `/exports`. The espresso example preserves the opening in the cup handle. It approximates the bowl of the spoon as a flat oval. [Photograph, project and OBJ](https://github.com/glitchykid/xtrifid/releases/download/v0.6.0/photo-example.zip)

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

The adult figure uses a MakeHuman anatomical mesh. The ears, lips, eyelids, fingers and torso belong to one continuous skin surface. The eyes have their own geometry. A plain person is unclothed. Request a cloak or hood explicitly. Select the body to adjust its proportions. The same material controls apply to the body and to each garment.

## Detail and available resources

Choose low, medium or high detail yourself. The human mesh has six resolutions across the two profiles. A project remembers your choice. Projects with PBR materials or saved UVs on baked cloth use XTRI format 5. Other saves use format 4. The studio reads formats 1–5. Basic figure prompts work offline. Free-form descriptions and text edits require Ollama weights.

Editable examples · [Person](https://github.com/glitchykid/xtrifid/releases/download/v0.5.0/human.xtri) · [Mouse shell](https://github.com/glitchykid/xtrifid/releases/download/v0.5.0/mouse-shell.xtri) · [Tabletop mirror](https://github.com/glitchykid/xtrifid/releases/download/v0.5.0/table-mirror.xtri) · [Coral](https://github.com/glitchykid/xtrifid/releases/download/v0.6.0/coral.xtri) · [Physics](https://github.com/glitchykid/xtrifid/releases/download/v0.6.0/physics-examples.zip) · [Textured vase](https://github.com/glitchykid/xtrifid/releases/download/v0.6.0/textured-vase.zip) · [Reference images and teapot](https://github.com/glitchykid/xtrifid/releases/download/v0.6.0/reference-example.zip)

Before preview or export, Xtrifid estimates triangle count against available RAM and Linux container limits. It reserves memory for the desktop and caps each scene at 150,000 triangles after triangulation. If the requested level exceeds the current budget, the previous scene stays available. Choose a lower level or Game Ready. When opening a saved project, you can accept a level that fits. Inference is limited separately by model size, memory, CPU threads and context length. Other applications can still consume system resources.

## Troubleshooting

- **The executable does not open.** Check execute permission, the graphical session and `ldd --version` for your glibc version. If Linux reports a missing `GLIBC` version, use the Docker instructions above.
- **The model download was interrupted.** Check the server’s free disk space and internet connection. A TLS error calls for checking the server’s trusted CA certificates. Then click **Скачать модель** again. A first CPU generation can take several minutes.
- **Docker** · **Black preview or `MESA: error: Failed to attach to x11 shm`.** In a Git checkout, run `git pull --ff-only` and restart `./studio`. For an archive installation, use version 0.2.1 or later. The launcher rebuilds the image with a Mesa driver that can display frames across the Docker IPC boundary.
- **Docker** · **No window.** Launch from the graphical session and check `echo "$DISPLAY"` and `xauth info`. The launcher forwards the current display cookie.
- **Docker** · **Ollama error or a slow first request.** Run `docker compose logs --tail 40 ollama` and `docker compose exec ollama ollama list`. A cold model takes time to load on CPU. The previous scene remains available after a failed request.
- **Unexpected geometry.** Specify quantities, units and contact points. Inspect the silhouette and mesh, then correct the selected part.

Report problems with the prompt, profile and reproduction steps. [GitHub Issues](https://github.com/glitchykid/xtrifid/issues) · [Architecture and Docker tests](docs/architecture.md)
