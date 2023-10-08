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

Download from our project page. Extract using:

```bash
python -m extraction.extract_schema_changes --help
```

### Run pipeline

```bash
./scripts/run_single.sh
```


## Development

### Run tests

```bash
pytest
```