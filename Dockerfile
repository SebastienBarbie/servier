FROM continuumio/anaconda3:latest
WORKDIR /root/servier

COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY model.py ./model.py
COPY main.py ./main.py
COPY dataset.py ./dataset.py
