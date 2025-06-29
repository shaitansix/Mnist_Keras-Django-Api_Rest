#!/usr/bin/env bash
set -o errexit

pyenv install $PYTHON_VERSION -s
pyenv global $PYTHON_VERSION

pip install -r requirements.txt
python manage.py collectstatic --no-input
python manage.py migrate