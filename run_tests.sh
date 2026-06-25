#!/bin/bash

# =============================================================================
# fix for script pathing [with source] [From SE#59895]
export OLD_PWD=${PWD}
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
   # if $SOURCE was a relative symlink, we need to resolve it relative to the
   # path where the symlink file was located
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
export DIR_PACKAGE=${DIR}

cd ${DIR_PACKAGE}
# =============================================================================

# =============================================================================

# export PYTHONPYCACHEPREFIX="/tmp/.pynalgo_cache/"

# test documentation
python -m pytest tests/
python -m mypy pynalgo/
# python -m pylint pynalgo/
python -m pytest --doctest-modules pynalgo/
# python -m mypy --strict pynalgo/

python -m ruff check pynalgo/
python -m ruff check tests/
python -m ruff check usage/

# python -m pytest --cov=tests/ --cov-report=term-missing

# =============================================================================

# =============================================================================
cd ${OLD_PWD}
# =============================================================================
