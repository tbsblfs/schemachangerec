#! /bin/bash

# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

# first argument is the path to the folder
folder=$1
redo=$2
projection=$3
# default projection is exclude
if [ -z "$projection" ]; then
    projection="exclude"
fi

embeddings="data/embeddings/enwiki_20180420_300d_entities.txt"
export TMPDIR=./tmp


build_graph() {
    strategy="$1"

    graph_option=""
    if [ "$strategy" == "baseline" ]; then
        graph_option="--strategy whole"
    else
        graph_option="--strategy $projection"
    fi

    graph_path="$folder/graph_$strategy"

    graph_build_cmd="python3 -m schemarec.mining.build_graph {} $graph_option --output $graph_path"
    echo "Building graph with command: $graph_build_cmd"
    # build graph
    find "$folder/train_graph/" -maxdepth 1 -name "*.json" | parallel --tmpdir tmp --eta -j16 "$graph_build_cmd"

    echo "Merging graph"
    python3 -m schemarec.mining.merge_graph \
        --nodes \
        "$graph_path"/*.nodes.json \
        >"$folder/$strategy-nodes.json"
    python3 -m schemarec.mining.merge_graph \
        "$graph_path"/*.edges.json \
        >"$folder/$strategy-edges.json"

    rm -rf "$graph_path"
}

overlap() {
    strategy="$1"
    split="$2"

    echo "Performing overlap on $split set for $strategy"
    python3 -m schemarec.overlap.overlap \
        --ignore-own-page \
        --reordering_strategy "$projection" \
        "$folder"/$split/*.json \
        --embedding $embeddings \
        --rules "$folder/rules-$strategy.pickle" \
        --output "$folder/overlap-$strategy-$split.pickle"
}

learn() {
    strategy="$1"
    learner="$2"

    echo "Learn ranking for $strategy"

    learn_option=""
    if [ "$learner" == "flaml" ]; then
        learn_option="--flaml"
    fi

    python3 -m schemarec.ranking.learn_ranking $learn_option \
        --input "$folder/overlap-$strategy-train_ranking.pickle" \
        --test "$folder/overlap-$strategy-test.pickle" \
        --output "$folder/ranking-$strategy-$learner.json"


    python3 -m schemarec.ranking.aggregate_results \
        --result "$folder/ranking-$strategy-$learner.json" \
        "$folder"/test/* >"$folder/ranking-$strategy-$learner-agg.json"
}

# clear times file
echo "" >"$folder/times.txt"

# iterate over "baseline" and "mining" as strategy
for strategy in "baseline" "lattice"; do
    echo "Processing $strategy"

    # if the graph already exists, skip building it (unless redo is graph or all)
    if [ -f "$folder/$strategy-nodes.json" ] && [ -f "$folder/$strategy-edges.json" ] && [ "$redo" != "graph" ] && [ "$redo" != "all" ]; then
        echo "Graph already exists, skipping"
    else
        graph_start=$(date +%s)
        build_graph $strategy
        graph_end=$(date +%s)
        echo "Graph building,$strategy,$((graph_end - graph_start))" >>"$folder/times.txt"
    fi

    if [ -f "$folder/rules-$strategy.pickle" ] && [ "$redo" != "rules" ] && [ "$redo" != "all" ]; then
        echo "Rules already exist, skipping"
    else
        rules_start=$(date +%s)
        python3 -m schemarec.mining.build_rule_df \
            --nodes "$folder/$strategy-nodes.json" \
            --edges "$folder/$strategy-edges.json" \
            --users "$folder/stats.json" \
            --output "$folder/rules-$strategy.pickle"
        rules_end=$(date +%s)
        echo "Rule building,$strategy,$((rules_end - rules_start))" >>"$folder/times.txt"
    fi

    for split in "train_ranking" "test"; do
        if [ -f "$folder/overlap-$strategy-$split.pickle" ] && [ "$redo" != "overlap" ] && [ "$redo" != "all" ]; then
            echo "Overlap already exists, skipping"
            continue
        fi
        overlap_start=$(date +%s)
        overlap "$strategy" "$split"
        overlap_end=$(date +%s)
        echo "Overlap $split,$strategy,$((overlap_end - overlap_start))" >>"$folder/times.txt"
    done

    if [ -f "$folder/ranking-$strategy-lgbm-agg.json" ] && [ "$redo" != "ranking" ]&& [ "$redo" != "all" ]; then
        echo "Ranking already exists, skipping"
        continue
    fi
    echo "Learn ranking for $strategy with lgbm"
    learn_start=$(date +%s)
    learn "$strategy" "lgbm"
    learn_end=$(date +%s)
    echo "Learning lgbm,$strategy,$((learn_end - learn_start))" >>"$folder/times.txt"


done
