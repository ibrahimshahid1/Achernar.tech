#!/bin/bash
set -e

# Wait for Postgres to be ready
echo "Waiting for PostgreSQL..."
while ! nc -z $DB_HOST ${DB_PORT:-5432}; do
  sleep 0.5
done
echo "PostgreSQL started"

# Run migrations
echo "Running migrations..."
python manage.py migrate --noinput

# Create superuser if it doesn't exist
echo "from django.contrib.auth import get_user_model; User = get_user_model(); User.objects.filter(username='admin').exists() or User.objects.create_superuser('admin', 'admin@achernar.io', 'admin')" | python manage.py shell

# Execute the command
exec "$@"
