###########
# GENERAL #
###########

# Starting from a base image (uv-based image)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install some essentials
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

####################
# PROJECT SPECIFIC #
####################

##
# Copy our project files into the container
#   ->  we only want the essential parts to keep 
#       our Docker image as small as possible

# COPY requirements.txt requirements.txt 
# not needed since we use uv and pyproject.toml
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY src/ src/
COPY data/ data/

# Copy the trained model (assumes it exists on host)
# This will need to be provided when running the container
COPY models/ models/

## 
# Install our project dependencies
# --locked      enforces strict adherence to uv.lock
# --no-cache    disables writing temporary download/wheel files to keep image size small

WORKDIR /

# Optimized version that uses caching of uv downloads/wheels between builds
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync

##
# The entry point is the application we want to run
# when the container starts up

ENTRYPOINT ["uv", "run", "src/ml_ops/evaluate.py"]

## Building the Docker image:
#
#     docker build -f dockerfiles/evaluate.dockerfile . -t evaluate:latest
#
#        -f dockerfiles/evaluate.dockerfile <-> specifies the Dockerfile to use
#        .                                   <-> the build context
#        -t evaluate:latest                  <-> tags the image with NAME "evaluate" and TAG "latest"
#
# Running the container:
#
#     docker run --name eval1 evaluate:latest models/model.pth
#
# Or mount the models directory if not included in the image:
#
#     docker run --name eval1 -v $(pwd)/models:/models evaluate:latest /models/model.pth
