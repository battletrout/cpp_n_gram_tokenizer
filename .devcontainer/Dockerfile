FROM mcr.microsoft.com/devcontainers/cpp:1-debian-11

# Install any additional packages you need
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    cmake \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*