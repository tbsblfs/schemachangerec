#! /bin/bash

# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT


export TMPDIR="$(pwd)/tmp"
mkdir -p data/splits

if [ ! -f data/tablekeys_test10.txt ]; then
    echo "Extract keys"
    find data/schemamatch -maxdepth 1 -name "*.json" | parallel --tmpdir tmp  --eta -j16 'jq -cr ".[0].key" {}' > data/tablekeys.txt
    total=$(wc -l < data/tablekeys.txt)
    shuf -n "$((total / 10))" data/tablekeys.txt > data/tablekeys_test10.txt
else
    echo "data/tablekeys_test10.txt already exists"
fi

find data/schemamatch/ -maxdepth 1 -name "*.json" | parallel --tmpdir tmp  --eta -j16 'python3 -m schemarec.extraction.train_test_split --keys data/tablekeys_test10.txt --output data/splits/ {}'

for split in data/splits/*; do
    # if split is not a directory, skip
    [ -d "$split" ] || continue

    echo "Processing $split"
    folder=$(basename "$split")
    find "$split/train" -maxdepth 1 -name "*.json" | parallel --tmpdir tmp  --eta -j16 "python3 -m  schemarec.stats.extract_stats --output $split/stats {}"
    python3 -m schemarec.stats.merge_stats "$split"/stats/*.json >"$split/stats.json"
    rm -rf "$split/stats/"



    if [ ! -f data/tablekeys_ranking.txt ]; then
        echo "Extract training keys"
        find "$split/train" -maxdepth 1 -name "*.json" | parallel --tmpdir tmp  --eta -j16 'jq -cr ".[0].key" {}' > "$split"/tablekeys_train.txt
        # extract 10% of tablekeys for ranking
        total=$(wc -l < "$split"/tablekeys_train.txt)
        shuf -n "$((total / 10))" "$split"/tablekeys_train.txt > "$split"/tablekeys_ranking.txt
    else
        echo "data/tablekeys_ranking.txt already exists"
    fi


    cmd="python3 -m schemarec.extraction.train_test_split --graphranking --keys $split/tablekeys_ranking.txt --output $split {}"

    find "$split/train" -maxdepth 1 -name "*.json" | parallel --tmpdir tmp  --eta -j16 "$cmd"
    mv "$split/spatiotemporal/train_graph" "$split/"
    mv "$split/spatiotemporal/train_ranking" "$split/"
    rm -rf "$split/spatiotemporal/"
done
