# Schema change recommendation

## Structure

Packages:
- ``extraction``: extract schema changes from matched tables, perform train/test split
- ``mining``: build rule graph
- ``overlap``: find matching rules
- ``ranking``: rank matching rules & evaluate
- ``schemamatching``: schema matching algorithm

## Usage

### Requirements
- Python 3.9
- Rust 1.72.0
- Jq 1.6
- Optional: GNU parallel

### Create a virtual env and install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Compile Rust code

```bash
cd rust-dist
maturin develop --release
```

### Download table data and extract schema changes

Download table changes from our project page. Extract schema changes using:

```bash
python -m extraction.extract_schema_changes --help
```

Or directly download extracted schema changes from our project page and unzip them to `data/schemamatch.

### Run train/test split

```bash
./scripts/create_splits.sh
```

This should create three folders: `data/splits/temporal`, `data/splits/spatial`and `data/splits/spatiotemporal`.

### Run pipeline

Download the embeddings file (300d txt) from [Wikipedia2Vec](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/) and unzip it.

Place the embeddings file `enwiki_20180420_300d_entities.txt` to `data/embeddings` and run the pipeline:

```bash
./scripts/run_all.sh
```

## Development

### Run tests

```bash
pytest
```