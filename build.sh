set -o errexit

pip install --upgrade pip
pip install -r requirements.txt
py manage.py makemigrations
py manage.py migrate