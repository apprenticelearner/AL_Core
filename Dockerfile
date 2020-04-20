FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

#RUN apt-get update && apt install -y software-properties-common && apt-get clean && rm -rf /var/lib/apt/lists/*
#RUN add-apt-repository ppa:deadsnakes/ppa

ARG APP_USER=appuser
RUN groupadd -r ${APP_USER} && useradd --no-log-init -r -g ${APP_USER} ${APP_USER}

# Install packages needed to run your application (not build deps):
#   mime-support -- for mime types when serving static files
#   postgresql-client -- for running database commands
# We need to recreate the /usr/share/man/man{1..8} directories first because
# they were clobbered by a parent image.
RUN set -ex \
    && RUN_DEPS=" \
    python3.8 \
    python3-pip \
    python3.8-dev \
    nano \
    git \
    libpcre3 \
    python3.8-dev \
    mime-support \
    postgresql-client \
    " \
    && seq 1 8 | xargs -I{} mkdir -p /usr/share/man/man{} \
    && apt-get update \
    && apt install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get install -y --no-install-recommends $RUN_DEPS \
    && rm -rf /var/lib/apt/lists/*

RUN python3.8 -m pip install --no-cache-dir torch
RUN python3.8 -m pip install --upgrade pip setuptools wheel

RUN mkdir -p /usr/local/apprentice
WORKDIR /usr/local/apprentice
COPY requirements.txt .

# Install build deps, then run `pip install`, then remove unneeded build deps all in a single step.
# Correct the path to your production requirements file, if needed.
RUN set -ex \
    && BUILD_DEPS=" \
    build-essential \
    libpcre3-dev \
    libpq-dev \
    " \
    && apt-get update && apt-get install -y --no-install-recommends $BUILD_DEPS \
    && python3.8 -m pip install --no-cache-dir -r requirements.txt --exists-action=w \
    && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false $BUILD_DEPS \
    && rm -rf /var/lib/apt/lists/*

COPY ./ /usr/local/apprentice

RUN python3.8 -m pip install --no-cache-dir .
RUN pytest tests

# uWSGI will listen on this port
EXPOSE 8000

# Add any static environment variables needed by Django or your settings file here:
ENV DJANGO_SETTINGS_MODULE=agent_api.settings

# Call collectstatic (customize the following line with the minimal environment variables needed for manage.py to run):
RUN DATABASE_URL='' python3.8 django/manage.py collectstatic --noinput

# Tell uWSGI where to find your wsgi file (change this):
ENV UWSGI_WSGI_FILE=./agent_api/wsgi.py

# Base uWSGI configuration (you shouldn't need to change these):
ENV UWSGI_HTTP=:8000 UWSGI_MASTER=1 UWSGI_HTTP_AUTO_CHUNKED=1 UWSGI_HTTP_KEEPALIVE=1 UWSGI_LAZY_APPS=1 UWSGI_WSGI_ENV_BEHAVIOR=holy

# Number of uWSGI workers and threads per worker (customize as needed):
ENV UWSGI_WORKERS=2 UWSGI_THREADS=4

# uWSGI static file serving configuration (customize or comment out if not needed):
# ENV UWSGI_STATIC_MAP="/static/=/code/static/" UWSGI_STATIC_EXPIRES_URI="/static/.*\.[a-f0-9]{12,}\.(css|js|png|jpg|jpeg|gif|ico|woff|ttf|otf|svg|scss|map|txt) 315360000"

# Deny invalid hosts before they get to Django (uncomment and change to your hostname(s)):
# ENV UWSGI_ROUTE_HOST="^(?!localhost:8000$) break:400"

# Change to a non-root user
USER ${APP_USER}:${APP_USER}

WORKDIR ./django 

# Start uWSGI
#CMD ["uwsgi", "--show-config"]