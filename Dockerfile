FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt install -y software-properties-common && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3-pip \
    git \
    nano \
    && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

RUN python3.8 -m pip install --upgrade pip setuptools wheel
RUN python3.8 -m pip install torch

RUN mkdir -p /usr/local/apprentice
WORKDIR /usr/local/apprentice
COPY ./ /usr/local/apprentice

RUN python3.8 -m pip install -r requirements.txt --exists-action=w
RUN python3.8 -m pip install .

RUN pytest tests

#CMD ["/usr/bin/python3.8", "/usr/local/apprentice/django/manage.py", "runserver"]
#CMD  ["/bin/sh", "-ec", "sleep 1000"]