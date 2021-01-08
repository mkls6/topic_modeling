# Demo
It's still unfinished, but you can run what is already done with main.py
entry point.
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