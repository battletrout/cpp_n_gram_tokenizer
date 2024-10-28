FROM mcr.microsoft.com/devcontainers/cpp:1-debian-11

    # Install Python, pip and other required packages
    RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
        && apt-get -y install --no-install-recommends \
        cmake \
        build-essential \
        python3 \
        python3-pip \
        python3-dev \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*
    
    # Install Python packages globally
    RUN python3 -m pip install --no-cache-dir \
        spacy \
        pybind11
    
    # Download English language model for spaCy
    RUN python3 -m spacy download en_core_web_sm