
# figure out which Python pyenv is using
PYTHON := $(shell pyenv which python3)

.PHONY: venv install clean

# create a local virtual environment in .venv
venv:
	$(PYTHON) -m venv .venv

# install requirements into the venv
install:
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

# freeze current venv packages into requirements.txt
update:
	. .venv/bin/activate && pip freeze > requirements.txt

# remove caches / build artifacts
clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache .venv