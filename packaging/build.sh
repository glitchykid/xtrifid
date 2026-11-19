#!/bin/sh
# Build the native distribution with the same Ollama release as the Compose service.
set -eu
cd "$(dirname -- "$0")/.."
ollama=$(awk '$1 == "image:" && $2 ~ /^ollama\/ollama:.*@sha256:/ { print $2 }' compose.yaml)
test -n "$ollama" || { echo 'Pinned Ollama image missing from compose.yaml' >&2; exit 1; }
output=exports/release
mkdir -p "$output"
staging=$(mktemp -d "$output/.native.XXXXXXXXXX")
trap 'rm -rf -- "$staging"' EXIT
trap 'exit 1' HUP INT TERM
docker build --platform linux/amd64 --file packaging/Dockerfile --target artifact \
    --build-arg "OLLAMA_IMAGE=$ollama" --output "type=local,dest=$staging" .
mv "$staging/xtrifid-linux-x86_64-notices.tar.gz" "$output/"
mv "$staging/xtrifid-linux-x86_64" "$output/"
