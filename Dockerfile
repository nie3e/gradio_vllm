FROM python:3.11-alpine

COPY requirements.txt /tmp/
RUN pip install --no-cache -r /tmp/requirements.txt

RUN mkdir -p /app
COPY src/ /app/
WORKDIR /app
RUN pip install -e /app/