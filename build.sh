set -o errexit

pip install -r requirements.txt
py manage.py makemigrations
py manage.py migrate