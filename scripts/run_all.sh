#! /bin/bash

redo="$1"

__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for split in "spatial" "temporal" "spatiotemporal"; do
      bash ${__dir}/run_single.sh "data/splits/$split" "$redo"
done