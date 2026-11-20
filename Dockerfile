FROM debian:trixie-slim@sha256:a72b4433de1cf7d6bae0c4cee74d0547f58601a643b3375f070c51600d63b6b2 AS base
# Mesa 26.1 probes whether the external X server can attach a client SHM segment.
# This signed snapshot pins the graphics stack; the normal Debian sources stay live.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && echo 'deb [signed-by=/usr/share/keyrings/debian-archive-keyring.pgp check-valid-until=no] https://snapshot.debian.org/archive/debian/20261104T212339Z/ trixie-backports main' > /etc/apt/sources.list.d/mesa.list \
    && apt-get update -o APT::Update::Error-Mode=any && apt-get install -y --no-install-recommends \
       fonts-dejavu-core gir1.2-gtk-4.0 python3-gi python3-gi-cairo python3-numpy python3-opengl \
       libegl-mesa0/trixie-backports libglx-mesa0/trixie-backports libgl1-mesa-dri/trixie-backports \
    && rm /etc/apt/sources.list.d/mesa.list && rm -rf /var/lib/apt/lists/*
RUN useradd --no-create-home --home-dir /tmp --uid 1000 studio
WORKDIR /app
ENV HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
FROM base AS runtime
COPY xtrifid ./xtrifid
USER 1000:1000
CMD ["python3", "-m", "xtrifid"]

FROM base AS test
COPY --from=ghcr.io/astral-sh/ruff:0.17.6@sha256:2671e9bf3e50c24a88e731f209f9bf01c0360da6857ae36bca680b222efc3528 /ruff /usr/local/bin/ruff
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb xauth dbus-daemon && install -d -m 1777 /tmp/.X11-unix && rm -rf /var/lib/apt/lists/*
COPY xtrifid ./xtrifid
COPY tests ./tests
COPY examples/art-studies/*.xtri ./examples/art-studies/
COPY pyproject.toml .
USER 1000:1000
ENV GTK_A11Y=none GDK_BACKEND=x11 LIBGL_ALWAYS_SOFTWARE=1
CMD ["sh", "-ec", "ruff check --no-cache . && ruff format --check --no-cache . && xvfb-run -a dbus-run-session -- python3 -m unittest discover -s tests -v"]
