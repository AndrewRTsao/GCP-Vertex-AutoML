# Top level build args
ARG build_for=linux/amd64

FROM --platform=$build_for python:3.8
SHELL ["/bin/bash", "-c"]

# Retrieving path of project and mirroring setup in container
ARG PROJECT_PATH
ENV LOCAL_PATH ${PROJECT_PATH}

RUN apt-get update \
  && apt-get dist-upgrade -y \
  && apt-get install -y --no-install-recommends \
    git \
    ssh-client \
    software-properties-common \
    make \
    build-essential \
    ca-certificates \
    libpq-dev \
  && apt-get clean \
  && rm -rf \
    /var/lib/apt/lists/* \
    /tmp/* \
    /var/tmp/*

# Copying files from git repo to docker container (and source env variables)
COPY . ${LOCAL_PATH}
WORKDIR ${LOCAL_PATH}

# Install dbt and other required packages for Vertex using requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel --no-cache-dir
RUN pip install --trusted-host pypip.python.org -r requirements.txt

# Changing work directory to dbt project
WORKDIR ${LOCAL_PATH}/dbt

CMD ["bash"]


