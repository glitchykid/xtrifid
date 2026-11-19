ARG OLLAMA_IMAGE=scratch
FROM ${OLLAMA_IMAGE} AS ollama
FROM ollama AS ollama_cpu
# Copy the directory as a tree so Docker preserves the original SONAME symlinks.
RUN mkdir /cpu && cp -a /usr/lib/ollama/*.so* /usr/lib/ollama/llama-server /cpu/
FROM ubuntu:24.04@sha256:73b999393d07f2bf65a40373778a44dc9a6d54ad7718d29247d8ff9413c4d6a8 AS ubuntu

FROM ubuntu AS toolchain
# Bootstrap HTTPS with Ubuntu's signed archive, then resolve the toolchain at one date.
RUN apt-get update -o APT::Update::Error-Mode=any && apt-get install -y --no-install-recommends ca-certificates \
    && sed -i -e 's|^URIs: .*|URIs: https://snapshot.ubuntu.com/ubuntu/20261118T000000Z/|' \
              -e '/^Types: deb$/a Check-Valid-Until: no' /etc/apt/sources.list.d/ubuntu.sources \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update -o APT::Update::Error-Mode=any && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        binutils ca-certificates gir1.2-gtk-4.0 libpython3.12 libgl1 \
        python3-gi python3-gi-cairo python3-numpy python3-opengl \
        python3-packaging python3-setuptools python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv --system-site-packages /opt/freezer
COPY packaging/freezer.txt /tmp/freezer.txt
RUN /opt/freezer/bin/pip install --require-hashes --only-binary=:all: --no-deps --no-cache-dir -r /tmp/freezer.txt

FROM toolchain AS build
WORKDIR /src
ENV PYTHONHASHSEED=0 SOURCE_DATE_EPOCH=0
COPY --from=ollama /usr/bin/ollama /opt/ollama/bin/ollama
COPY --from=ollama_cpu /cpu/ /opt/ollama/lib/ollama/
COPY --from=ollama /usr/lib/ollama/*LICENSE* /usr/lib/ollama/*NOTICE* /opt/ollama/licenses/
COPY xtrifid ./xtrifid
COPY packaging/launch.py packaging/notices.py packaging/xtrifid.spec packaging/OLLAMA-LICENSE ./packaging/
RUN PATH="/usr/lib/x86_64-linux-gnu/glib-2.0:$PATH" /opt/freezer/bin/pyinstaller --noconfirm packaging/xtrifid.spec \
    && /opt/freezer/bin/python packaging/notices.py
COPY packaging/check_abi.py ./packaging/
RUN /opt/freezer/bin/python packaging/check_abi.py dist/xtrifid-linux-x86_64

FROM scratch AS artifact
COPY --from=build /src/dist/xtrifid-linux-x86_64 /
COPY --from=build /src/dist/xtrifid-linux-x86_64-notices.tar.gz /

FROM ubuntu AS smoke
RUN apt-get update -o APT::Update::Error-Mode=any && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        dbus-daemon fonts-dejavu-core libgl1-mesa-dri libglx-mesa0 libegl1 libegl-mesa0 xauth xdotool xvfb \
    && install -d -m 1777 /tmp/.X11-unix && rm -rf /var/lib/apt/lists/*
COPY packaging/smoke.sh /smoke.sh
USER 1000:1000
ENV GTK_A11Y=none GDK_BACKEND=x11 LIBGL_ALWAYS_SOFTWARE=1
ENTRYPOINT ["xvfb-run", "-a", "dbus-run-session", "--", "sh", "/smoke.sh"]
