#!/bin/bash
# Run on the submission/login node, without root or a GPU.
set -euo pipefail
deepswe_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[[ "$(uname -m)" == x86_64 ]] || { echo 'This uv artifact requires x86_64' >&2; exit 1; }
destination="$deepswe_root/.cache/deepswe-tools/uv-0.9.7"
if [[ -x "$destination/uv" ]] && [[ "$("$destination/uv" --version)" == 'uv 0.9.7' ]]; then
    exit 0
fi
mkdir -p "$(dirname "$destination")"
temporary="$(mktemp -d "$(dirname "$destination")/uv-download.XXXXXX")"
trap 'rm -rf -- "$temporary"' EXIT
curl --fail --location --silent --show-error --retry 3 --max-time 120 \
    https://github.com/astral-sh/uv/releases/download/0.9.7/uv-x86_64-unknown-linux-gnu.tar.gz \
    --output "$temporary/uv.tar.gz"
printf '%s  %s\n' b26fcc8dfa1c39b5a5613445af3be3eefda45d9a39359bee271eafe34913583e \
    "$temporary/uv.tar.gz" | sha256sum --check --status
tar -xzf "$temporary/uv.tar.gz" -C "$temporary"
mkdir -p "$destination"
install -m 755 "$temporary/uv-x86_64-unknown-linux-gnu/uv" "$temporary/uv"
"$temporary/uv" --version
mv -f "$temporary/uv" "$destination/uv"
