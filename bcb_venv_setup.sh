#!/bin/bash
# setup_bigcodebench_env.sh
# Run once to create the BigCodeBench virtual environment.
# Usage: bash bcb_venv_setup.sh

set -e

# load .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

PYTHON_BIN=${PYTHON_BIN:-python3.10}
VENV_DIR=${BCB_VENV_DIR:-~/.bigcodebench_venv}

echo "[setup] Using Python: $PYTHON_BIN"
echo "[setup] Venv dir: $VENV_DIR"

# check python version
$PYTHON_BIN -c "import sys; assert sys.version_info >= (3,10), f'Python 3.10+ required, got {sys.version}'"

# create venv
$PYTHON_BIN -m venv "$VENV_DIR"

# install dependencies
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install pytest-cov pytest-json-report matplotlib openpyxl scikit-learn flask faker bs4 natsort xlwt scikit-image requests_mock django texttable nltk rsa pycryptodome seaborn TextBlob wikipedia flask_restful pyquery mechanize wordcloud statsmodels xlrd flask-mail xmltodict
"$VENV_DIR/bin/pip" install matplotlib openpyxl
echo "[setup] Done. Venv ready at $VENV_DIR"