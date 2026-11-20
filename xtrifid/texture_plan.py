"""Translate a free description into a bounded local diffuse material recipe."""

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
_INSTRUCTIONS = """Interpret the user’s material description and return a single procedural diffuse-colour recipe.
Choose the closest underlying surface: directional wood grain, mineral stone, woven fabric,
porous skin/leather, brushed metal or smooth paint (also ceramics, enamel, plastic, glass or paper).
color_a and color_b are the darker and lighter tones of the base. Select a repeating motif of
spots, stripes or fish-like overlapping scales if the description calls for one; otherwise choose none.
Use the accent tone for the motif. Encode colours as six hexadecimal sRGB digits preceded by #.
Scale is pattern frequency: 1 is normal, smaller values produce fewer, larger marks.
Preserve named colours and the requested visual effect as closely as this recipe allows."""


def text_to_texture(planner, prompt, seed=0, resolution=256):
    """Run on the Studio worker thread. Only the caller can set seed and pixel budget."""
    TextureSpec.parse({"material": "paint", "seed": seed, "resolution": resolution})
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 500:
        raise ValueError("Введите описание текстуры длиной до 500 символов")

    def parse(recipe):
        if (
            not isinstance(recipe, dict)
            or recipe.keys() != _FIELDS.keys()
            or recipe["accent"] is None
        ):
            raise ValueError("Модель вернула неполный рецепт текстуры")
        return TextureSpec.parse({**recipe, "seed": seed, "resolution": resolution})

    return planner._plan(
        prompt, "", _SCHEMA, parse, instructions=_INSTRUCTIONS, num_ctx=2048, num_predict=320
    )
