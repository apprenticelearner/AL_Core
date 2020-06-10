#! /usr/bin/env bash
sleep 5; 
# Run migrations 
python3.8 manage.py migrate
uwsgi --show-config
