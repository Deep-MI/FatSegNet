## start with the Docker 'base Tensorflow1.4 for python3' Debian-based image

#FROM tensorflow/tensorflow:1.6.0-py3
FROM tensorflow/tensorflow:1.6.0-gpu-py3
#FROM tensorflow/tensorflow:latest-gpu-py3
##Install custom libraries

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-tk

# install dependencies from python packages
RUN pip3 --no-cache-dir install \
    pandas==0.21.0\
    scikit-learn==0.19.1 \
    scipy==1.1.0\
    scikit-image==0.15.0 \
    SimpleITK==1.1.0 \
    nibabel==2.2.1 \
    keras==2.2.4 \
    numpy==1.15.4





## Copying application code (present working directory) and configuration to docker image
COPY ./tool /tool
WORKDIR "/tool"
RUN bash /tool/bash_profile


ENTRYPOINT ["python3","./run_FatSegNet.py"]

