#!/bin/bash

cd /django_app

echo "Starting migrations..."
python manage.py makemigrations
python manage.py migrate

echo "Collecting static files..."
python manage.py collectstatic --noinput --clear

echo "Starting Gunicorn..."
exec gunicorn EULA_VAAAGm.wsgi:application --bind 0.0.0.0:8001 --workers 8 --log-level debug --chdir /django_app
