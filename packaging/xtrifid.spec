a = Analysis(
    ["launch.py"],
    pathex=["/src"],
    binaries=[
        ("/opt/ollama/bin/ollama", "ollama/bin"),
        ("/opt/ollama/lib/ollama/*.so*", "ollama/lib/ollama"),
        ("/opt/ollama/lib/ollama/llama-server", "ollama/lib/ollama"),
        ("/usr/lib/x86_64-linux-gnu/libwayland-egl.so.1", "."),
        ("/usr/lib/x86_64-linux-gnu/libOpenGL.so.0", "."),
        ("/usr/lib/x86_64-linux-gnu/libGLESv2.so.2", "."),
    ],
    datas=[
        ("/opt/ollama/licenses/*", "ollama/licenses"),
        ("OLLAMA-LICENSE", "ollama/licenses"),
        ("/etc/fonts/fonts.conf", "fontconfig"),
    ],
    hiddenimports=[
        "cairo",  # gi._gi_cairo needs the Pycairo C API for Gtk.DrawingArea callbacks.
        "gi.repository.Gtk",
        "gi.repository.Gdk",
        "gi.repository.Gsk",
        "OpenGL.platform.egl",
    ],
    hooksconfig={
        "gi": {
            "module-versions": {"Gtk": "4.0", "Gdk": "4.0"},
            "icons": ["Adwaita"],
            "themes": ["Adwaita"],
            "languages": ["ru", "en_GB"],
        },
    },
)
exe = EXE(
    PYZ(a.pure),
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="xtrifid-linux-x86_64",
    console=True,
    upx=False,
)
