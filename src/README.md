# Demo
It's still unfinished, but you can run what is already done with `main.py`
entry point.
## Environment
This project uses Conda for dependency management.
Restore the env using following command:

```
conda env create -f environment.yml
conda activate topic_modeling
```

Next, you'll need to download and install Spacy language models:
```
python -m spacy download en_core_web_lg
python -m spacy download ru_core_news_lg
```

## Run
* Add pkg to `PYTHONPATH` variable:
```shell
# Bash or any other POSIX-compliant shell:
# Assuming you already cd'd to src dir
export PYTHONPATH=${PYTHONPATH:+${PYTHONPATH}:}$PWD/pkg
# fish
set -ax PYTHONPATH $PWD/pkg
```
* Run `main.py`
```shell
python3 main.py
```
* * Alternatively, add executable bit `chmod +x main.py` and run `./main.py`
    
## Run tests
1. Update `PYTHONPATH` as in example above
1. `pytest ./tests`