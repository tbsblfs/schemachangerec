#! /bin/bash

# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT


export TMPDIR=/home/tobias.bleifuss/python/schemachangerules/tmp
mkdir -p data/splits

find data/schemamatch -maxdepth 1 -name "*.json" | parallel --tmpdir tmp  --eta -j16 'jq -cr ".[0].key" {}' > data/splits/tablekeys.txt
total=$(wc -l < data/splits/tablekeys.txt)
shuf -n "$((total / 10))" data/splits/tablekeys.txt > data/splits/tablekeys_test10.txt

find data/schemamatch/ -maxdepth 1 -name "*.json" | parallel --tmpdir tmp  --eta -j16 'python3 -m extraction.train_test_split --keys data/splits/tablekeys_test10.txt --output data/splits/ {}'

for split in data/splits/*; do
    # if split is not a directory, skip
    [ -d "$split" ] || continue

    echo "Processing $split"
    folder=$(basename "$split")
    find "$split/train" -maxdepth 1 -name "*.json" | parallel --tmpdir tmp  --eta -j16 "python3 -m  analysis.lattice.construct.stats.extract_stats --output $split/stats {}"
    python3 -m schemarec.mining.construct.stats.merge_stats "$split"/stats/*.json >"$split/stats.json"
    rm -rf "$split/stats/"

    find "$split/train" -maxdepth 1 -name "*.json" | parallel --tmpdir tmp  --eta -j16 'jq -cr ".[0].key" {}' > "$split"/tablekeys_train.txt

    # extract 10% of tablekeys for ranking
    total=$(wc -l < "$split"/tablekeys_train.txt)
    shuf -n "$((total / 10))" "$split"/tablekeys_train.txt > "$split"/tablekeys_ranking.txt

    cmd="python3 -m extraction.train_test_split --graphranking --keys $split/tablekeys_ranking.txt --output $split {}"

    find "$split/train" -maxdepth 1 -name "*.json" | parallel --tmpdir tmp  --eta -j16 "$cmd"
    
    for subsplit in "random" "spatial" "temporal" "tempospatial"; do
        ln -s "$(pwd)/$split/test" "$split/$subsplit/test"
    done
done
