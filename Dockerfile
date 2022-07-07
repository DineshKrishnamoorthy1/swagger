FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt install software-properties-common -y
RUN apt-get -y install python3-pip
RUN apt-get install ffmpeg libsm6 libxext6 tesseract-ocr -y
RUN apt-get install poppler-utils -y
#RUN /usr/local/bin/python -m pip install --upgrade pip

ENV PYTHONUNBUFFERED 1

RUN mkdir /app

WORKDIR /app

COPY pyproject.toml /app/

#RUN python3 -V

RUN pip3 install poetry

RUN poetry config virtualenvs.in-project true
RUN poetry install

COPY . .

RUN pwd

RUN ls -ltr
