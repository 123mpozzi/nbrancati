FROM ubuntu:xenial

RUN apt update && apt install -y libopencv-dev

COPY install-opencv.sh /install-opencv.sh

RUN apt install -y wget unzip libqt4-dev

RUN chmod +x /install-opencv.sh && /install-opencv.sh

WORKDIR /app