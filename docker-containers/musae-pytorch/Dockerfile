#FROM python:3.6
FROM nvidia/cuda:11.3.0-base-ubuntu18.04
#CMD nvidia-smi

# Installs necessary dependencies.
RUN apt-get update 
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update 

RUN apt-get install -y --no-install-recommends \
         wget \
         curl \
         gcc \
         python-pydot \
         python-pydot-ng \
         graphviz \
         python3.8-dev \
         python3.8-distutils

RUN rm -rf /var/lib/apt/lists/*

# Installs pip.
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    pip install setuptools && \
    rm get-pip.py

RUN pip install virtualenv
ENV VIRTUAL_ENV=/pytorch-gpu
ENV VIRTUAL_BIN=$VIRTUAL_ENV/bin
RUN virtualenv /pytorch-gpu -p python3.8
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN $VIRTUAL_BIN/python -m pip install --upgrade pip

# RUN . $VIRTUAL_ENV/bin/activate

# INSTALL DEPENDENCIES
WORKDIR /app
ADD ./requirements.txt /app/requirements.txt
RUN $VIRTUAL_BIN/pip install -r requirements.txt

# Jupyter env gpu
RUN /pytorch-gpu/bin/python -m ipykernel install --user --name=pytorch-gpu

# CPU friendly env
ENV VIRTUAL_ENV=/pytorch-cpu
ENV VIRTUAL_BIN=$VIRTUAL_ENV/bin
RUN virtualenv /pytorch-cpu -p python3.8
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# RUN . "$VIRTUAL_ENV/bin/activate"

RUN $VIRTUAL_BIN/python -m pip install --upgrade pip

# INSTALL DEPENDENCIES
WORKDIR /app
ADD ./requirements-cpu.txt /app/requirements-cpu.txt
RUN $VIRTUAL_BIN/pip install -r requirements-cpu.txt

# Jupyter env gpu
RUN /pytorch-cpu/bin/python -m ipykernel install --user --name=pytorch-cpu


# COPY FILES
ADD ./app /app

WORKDIR /home
ADD ./home /home

# Launch Jupyter
EXPOSE 8888
ENTRYPOINT  ["/pytorch-gpu/bin/jupyter", "notebook", "--port=8888", "--no-browser", "--ip=*", "--allow-root", "--NotebookApp.token=''", "--notebook-dir=/home/notebooks"]

## docker build -t musae-orig-gpu .
## docker run -dp 8888:8888 -w /app -v "$(pwd)/app:/app" -v "$(pwd)/home:/home" musae-gpu-orig