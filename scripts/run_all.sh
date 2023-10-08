#! /bin/bash

redo="$1"

for split in "spatial" "temporal" "tempospatial"; do
    for substrategy in "spatial" "temporal" "tempospatial" "random"; do
        ./run_single.sh "data/splits/$split/$substrategy" "$redo"
    done
done