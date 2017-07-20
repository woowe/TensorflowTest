FROM gcr.io/tensorflow/tensorflow:latest-gpu

RUN pip install keras
RUN pip install h5py