#!/bin/sh
# Prepare the smoke image and cache qwen3.5:4b, then run from the checkout:
# docker build -f packaging/Dockerfile --target smoke -t xtrifid-native-smoke .
# sh tests/native_offline.sh exports/release/xtrifid-linux-x86_64 [weights-volume]
set -eu

if [ "${1:-}" != --inside ]; then
    binary=$(realpath "${1:?Provide the release executable}")
    volume=${2:-xtrifid_weights}
    test -x "$binary"
    docker volume inspect "$volume" >/dev/null
    exec docker run --rm --init --network none --read-only --cap-drop ALL \
        --security-opt no-new-privileges --tmpfs /tmp:exec,mode=1777 \
        -v "$binary:/app/xtrifid:ro" -v "$(realpath "$0"):/probe.sh:ro" \
        -v "$volume:/weights:ro" -e OLLAMA_MODELS=/weights/models \
        -e GSETTINGS_BACKEND=memory -e LC_ALL=C.UTF-8 --entrypoint xvfb-run \
        xtrifid-native-smoke -a -s '-screen 0 1280x900x24' dbus-run-session -- sh /probe.sh --inside
fi

for tool in python3 docker git; do
    if command -v "$tool" >/dev/null; then exit 1; fi
done
test ! -e /var/run/docker.sock
if ldconfig -p | grep -q libgtk-4; then exit 1; fi
export HOME=/tmp/home XDG_RUNTIME_DIR=/tmp/runtime XTRIFID_FILES=/tmp/exports
mkdir -m 700 "$HOME" "$XDG_RUNTIME_DIR" "$XTRIFID_FILES" '/tmp/working directory'
cd '/tmp/working directory'
/app/xtrifid > /tmp/client.log 2>&1 &
app=$!
trap 'cat /tmp/client.log >&2' EXIT
window=$(timeout 20 xdotool search --sync --onlyvisible --name '^Xtrifid$' | head -1)
test -n "$window"
xdotool windowfocus --sync "$window"
# Coordinates are in the actual GTK window on the fixed-size private display.
xdotool mousemove --window "$window" 140 109 click 1
prompt='Создай один прямоугольный брусок без других деталей. Размер по X 40 см, по Y 20 см, по Z 10 см. Без поворота, нижняя грань лежит на полу.'
printf %s "$prompt" | xclip -selection clipboard -in
xdotool key --clearmodifiers ctrl+v ctrl+a
step=0
until [ "$(timeout 1 xclip -selection primary -out 2>/dev/null || :)" = "$prompt" ]; do
    step=$((step + 1))
    [ "$step" -lt 30 ] || { echo 'GTK did not receive the UTF-8 prompt' >&2; exit 1; }
    xdotool key --clearmodifiers ctrl+a
    sleep 0.1
done
echo "Verified the UTF-8 prompt in the GTK entry."
xdotool key Return

# The export button becomes sensitive when generation creates the first scene.
step=0
until dialog=$(xdotool search --onlyvisible --name '^Экспорт OBJ$' | head -1) && [ -n "$dialog" ]; do
    kill -0 "$app"
    step=$((step + 1))
    [ "$step" -lt 350 ] || { echo 'Timed out generating the model' >&2; exit 1; }
    xdotool mousemove --window "$window" 958 28 click 1
    sleep 1
done
xdotool windowfocus --sync "$dialog" key Return
step=0
until [ -s "$XTRIFID_FILES/model.obj" ]; do
    step=$((step + 1))
    [ "$step" -lt 60 ] || { echo 'Timed out saving OBJ' >&2; exit 1; }
    sleep 0.2
done

awk '
    $1=="o" { print "OBJ object:", $2 }
    $1=="v" {
        for (i=2; i<=4; i++) {
            if (!v || $i<lo[i]) lo[i]=$i
            if (!v || $i>hi[i]) hi[i]=$i
        }
        v++
    }
    $1=="f" { f++; if (NF!=4) bad=1 }
    END {
        print "Offline OBJ:", v, "vertices,", f, "faces;", hi[2]-lo[2], hi[3]-lo[3], hi[4]-lo[4], "metres"
        exit (v!=8 || f!=12 || bad || hi[2]-lo[2]!=0.4 || hi[3]-lo[3]!=0.2 || hi[4]-lo[4]!=0.1 || lo[4]!=0)
    }
' "$XTRIFID_FILES/model.obj"

children=
for path in /proc/[0-9]*/exe; do
    case "$(readlink "$path")" in
        /tmp/_MEI*/ollama/*) children="$children ${path%/exe}" ;;
    esac
done
test -n "$children"
kill -TERM "$app"
wait "$app"
for child in $children; do
    [ ! -e "$child/exe" ] || { echo "Ollama process survived shutdown at $child" >&2; exit 1; }
done
echo 'The packaged CPU server and runner exited.'
