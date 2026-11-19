#!/bin/sh
# Run after 'docker compose --profile dev build studio tests'. Optional args are image tags.
set -eu
tests=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
state=$(mktemp -d "${TMPDIR:-/tmp}/xtrifid-x11.XXXXXXXXXX")
server="xtrifid-xvfb-${state##*/}"
client="xtrifid-client-${state##*/}"
cleanup() {
    docker rm -f "$client" "$server" >/dev/null 2>&1 || :
    rm -rf -- "$state"
}
trap cleanup EXIT
trap 'exit 1' INT TERM HUP
mkdir -m 1777 "$state/socket"
touch "$state/authority"
identity="$(id -u):$(id -g)"
docker run -d --init --name "$server" --network none --ipc private --hostname xtrifid-test \
    --user "$identity" -v "$state:/display" -v "$state/socket:/tmp/.X11-unix" \
    "${2:-xtrifid-tests}" xvfb-run -e /dev/stderr -n 99 -f /display/authority \
    -s '-screen 0 1280x900x24 -fbdir /display' \
    sh -ec 'touch /display/ready; exec sleep infinity' >/dev/null
tries=0
until [ -f "$state/ready" ]; do
    if [ "$tries" -ge 100 ]; then
        echo 'Timed out starting Xvfb' >&2
        docker logs "$server" >&2
        exit 1
    fi
    tries=$((tries + 1))
    sleep 0.1
done
result=0
docker run --rm --name "$client" --network none --ipc private --hostname xtrifid-test \
    --user "$identity" -v "$state:/display:ro" -v "$state/socket:/tmp/.X11-unix:ro" \
    -v "$tests:/probe:ro" -e XAUTHORITY=/display/authority -e DISPLAY=:99 \
    -e GDK_BACKEND=x11 -e GTK_A11Y=none -e GDK_SCALE=1 -e PYTHONPATH=/app \
    "${1:-xtrifid-studio}" python3 /probe/external_x11.py 2>"$state/stderr" || result=$?
awk '/Failed to attach to x11 shm/ { if (!count++) print; next } { print }
     END { if (count) { print count " Mesa X11 SHM errors"; exit 1 } }' \
    "$state/stderr" >&2 || result=1
exit "$result"
