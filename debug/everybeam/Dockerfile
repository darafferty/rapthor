FROM python:3.9

RUN apt-get update && \
    apt-get install -y \
        gdb \
        less \
        strace \
        vim

RUN groupadd \
    -g 1001 \
    rapthor

RUN useradd \
    -g 1001 \
    -m \
    -s /bin/bash \
    -u 15511 \
    rapthor

USER rapthor
WORKDIR /home/rapthor

RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install git+https://git.astron.nl/RD/rapthor.git@RAP-296_invalid-pointer-when-loading-skymodel
