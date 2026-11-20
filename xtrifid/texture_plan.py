"""Translate a free description into a bounded local surface recipe."""

from .textures import MOTIFS, PALETTES, TextureSpec

COLOR = {"type": "string", "pattern": "^#[0-9a-fA-F]{6}$"}
_FIELDS = {
    "material": {"type": "string", "enum": list(PALETTES)},
    "motif": {"type": "string", "enum": list(MOTIFS)},
    "color_a": COLOR,
    "color_b": COLOR,
    "accent": COLOR,
    "scale": {"type": "number", "minimum": 0.25, "maximum": 8},
}
_SCHEMA = {
    "type": "object",
    "properties": _FIELDS,
    "required": list(_FIELDS),
    "additionalProperties": False,
}
_PHYSICAL_FIELDS = {
    name: {"type": "number", "minimum": 0, "maximum": upper}
    for name, upper in (("roughness", 1), ("metallic", 1), ("normal_strength", 2))
}
_PBR = {
    "type": "object",
    "properties": _PHYSICAL_FIELDS,
    "required": list(_PHYSICAL_FIELDS),
    "additionalProperties": False,
}
_PHYSICAL_SCHEMA = {
    **_SCHEMA,
    "properties": {**_FIELDS, "pbr": _PBR},
    "required": [*_FIELDS, "pbr"],
}
_INSTRUCTIONS = """Interpret the user’s material description and return a single procedural diffuse-colour recipe.
Choose the closest underlying surface: directional wood grain, mineral stone, woven fabric,
porous skin/leather, brushed metal or smooth paint (also ceramics, enamel, plastic, glass or paper).
color_a and color_b are the darker and lighter tones of the base. Select a repeating motif of
spots, stripes or fish-like overlapping scales if the description calls for one; otherwise choose none.
Use the accent tone for the motif. Encode colours as six hexadecimal sRGB digits preceded by #.
Scale is pattern frequency: 1 is normal, smaller values produce fewer, larger marks.
Preserve named colours and the requested visual effect as closely as this recipe allows."""


def text_to_texture(planner, prompt, seed=0, resolution=256, *, physical=False):
    """Run on the Studio worker thread. Only the caller can set seed and pixel budget."""
    TextureSpec.parse({"material": "paint", "seed": seed, "resolution": resolution})
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 500:
        raise ValueError("Введите описание текстуры длиной до 500 символов")

    schema = _PHYSICAL_SCHEMA if physical else _SCHEMA
    instructions = _INSTRUCTIONS
    if physical:
        instructions += """
Also select the surface finish in pbr. roughness ranges from 0 (polished) to 1 (matte).
metallic is 1 for bare metal, 0 for nonmetals including paint over metal.
normal_strength is baked microsurface relief: 0 for completely flat, 1 for ordinary relief,
up to 2 for pronounced relief. Respect explicit finish and strength requests."""

    def parse(recipe):
        if (
            not isinstance(recipe, dict)
            or recipe.keys() != schema["properties"].keys()
            or recipe["accent"] is None
            or (physical and not isinstance(recipe["pbr"], dict))
        ):
            raise ValueError("Модель вернула неполный рецепт текстуры")
        return TextureSpec.parse({**recipe, "seed": seed, "resolution": resolution})

    return planner._plan(
        prompt,
        "",
        schema,
        parse,
        instructions=instructions,
        num_ctx=2048,
        num_predict=384 if physical else 320,
    )
