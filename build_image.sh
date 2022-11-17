#!/bin/bash

# Set docker image name
IMAGE="vertex-${DOCKER_REPO}-dbt:latest"

# Create repo in the artifact registry
gcloud beta artifacts repositories create ${DOCKER_REPO} \
 --repository-format=docker \
 --project=${GOOGLE_CLOUD_PROJECT} --location=${VERTEX_REGION}

# Configure Docker
gcloud auth configure-docker ${VERTEX_REGION}-docker.pkg.dev

# Build and push docker image
docker build --build-arg PROJECT_PATH=${LOCAL_PROJECT} --tag=${VERTEX_REGION}-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT}/${DOCKER_REPO}/${IMAGE} .
docker push ${VERTEX_REGION}-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT}/${DOCKER_REPO}/${IMAGE}

# Retrieve docker image name
export VERTEX_DBT_DOCKER_IMAGE=$(docker inspect --format="{{index .RepoDigests 0}}" ${VERTEX_REGION}-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT}/${DOCKER_REPO}/${IMAGE})
echo ${VERTEX_DBT_DOCKER_IMAGE}