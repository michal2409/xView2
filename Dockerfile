ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.03-py3
FROM ${FROM_IMAGE_NAME}

ADD . /workspace/xview2
WORKDIR /workspace/xview2

RUN pip install --upgrade pip
RUN pip install --disable-pip-version-check -r requirements.txt