FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04


# Make sure some default required packages are installed into venv
RUN apt-get update && apt-get install -y python3-pip git
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.8

RUN python3.8 -m pip install --upgrade pip setuptools wheel
RUN python3.8 -m pip install torch

RUN git clone http://gitlab+deploy-token-9:19fecH2MzpmrGG2LULts@hq-git.soartech.com/apprentice/apprentice.git
WORKDIR /apprentice
RUN python3.8 setup.py install

RUN python3.8 -m pip install -U pytest #remove when pytest added to requirements
RUN python3.8 -m pip install multiprocess
RUN python3.8 -m pip install -r requirements.txt -I --exists-action=w

RUN pytest tests

# Configure Services and Port
COPY start.sh ./start.sh
CMD ["./start.sh"]

#EXPOSE 8000 #django default
