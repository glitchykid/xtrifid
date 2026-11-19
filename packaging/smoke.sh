#!/bin/sh
# Run in the smoke stage with an X server and the binary mounted at /artifact.
set -eu
binary=${1:-/artifact/xtrifid-linux-x86_64}
test -x "$binary" || { echo "Missing executable: $binary" >&2; exit 1; }
for tool in docker git python3; do
    if command -v "$tool" >/dev/null; then echo "Unexpected host tool: $tool" >&2; exit 1; fi
done
if ldconfig -p | grep -q 'libgtk-4'; then echo 'Unexpected system GTK' >&2; exit 1; fi
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
mkdir "$work/empty" "$work/home" "$work/runtime"
export HOME="$work/home" TMPDIR="$work/runtime" GTK_IM_MODULE=ibus GTK_DEBUG=modules
cd "$work/empty"
"$binary" >"$work/log" 2>&1 &
pid=$!
server=
cleanup() {
    status=$?
    if [ -n "$server" ]; then
        kill "$server" 2>/dev/null || :
        wait "$server" 2>/dev/null || :
    fi
    kill "$pid" 2>/dev/null || :
    wait "$pid" 2>/dev/null || :
    if [ "$status" -ne 0 ]; then cat "$work/log" >&2; fi
    rm -rf -- "$work"
}
trap cleanup EXIT
tries=0
until window=$(xdotool search --onlyvisible --name '^Xtrifid$'); do
    kill -0 "$pid" && [ "$tries" -lt 400 ] || exit 1
    tries=$((tries + 1))
    sleep 0.1
done
xdotool windowfocus --sync "$window" mousemove --window "$window" 140 104 click 1
xdotool type 'mesh preview'
xdotool key ctrl+a ctrl+c
tries=0
until [ "$(timeout 2 xclip -selection clipboard -out 2>/dev/null)" = 'mesh preview' ]; do
    [ "$tries" -lt 30 ] || { echo 'Model description did not accept input' >&2; exit 1; }
    tries=$((tries + 1))
    sleep 0.1
done
if grep -Fq 'No IM module matching GTK_IM_MODULE=ibus' "$work/log"; then exit 1; fi
grep '^extension: ibus:' "$work/log"
test -z "$(ls -A)"
echo 'GTK window displayed from an empty directory without a system Python or GTK.'

set -- "$TMPDIR"/_MEI*/ollama/bin/ollama
test "$#" -eq 1 && test -x "$1"
libdir="$(dirname "$1")/../lib/ollama"
test -L "$libdir/libggml-base.so"
"$libdir/llama-server" --version
export OLLAMA_HOST=127.0.0.1:25434 OLLAMA_NO_CLOUD=1
"$1" serve >"$work/ollama.log" 2>&1 &
server=$!
tries=0
until grep -q 'library=cpu' "$work/ollama.log"; do
    if ! kill -0 "$server" || [ "$tries" -ge 100 ]; then cat "$work/ollama.log" >&2; exit 1; fi
    tries=$((tries + 1))
    sleep 0.1
done
"$1" list
echo 'Bundled Ollama served its model list using the CPU backend.'
