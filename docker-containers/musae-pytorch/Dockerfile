#FROM python:3.6
FROM nvidia/cuda:11.3.0-base-ubuntu18.04

#CMD nvidia-smi

# Installs necessary dependencies.
RUN apt-get update 

RUN apt-get install -y --no-install-recommends \
         wget \
         curl \
         gcc \
         python-pydot \
         python-pydot-ng \
         graphviz \
         python-dev

RUN rm -rf /var/lib/apt/lists/*

# Installs pip.
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    pip install setuptools && \
    rm get-pip.py

RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3.8
ENV PATH="$VIRTUAL_ENV/bin:$PATH"


# INSTALL DEPENDENCIES
WORKDIR /app
ADD ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt


# COPY FILES
ADD ./app /app

WORKDIR /home
ADD ./home /home

# Launch Jupyter
EXPOSE 8888
ENTRYPOINT  ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=*", "--allow-root", "--NotebookApp.token=''", "--notebook-dir=/home/notebooks"]

## docker build -t musae-orig-gpu .
## docker run -dp 8888:8888 -w /app -v "$(pwd)/app:/app" -v "$(pwd)/home:/home" musae-gpu-orig