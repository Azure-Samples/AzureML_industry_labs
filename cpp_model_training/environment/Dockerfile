FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04

# Install C++ compiler
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

# Compile C++ training binary
COPY src/train.cpp /tmp/train.cpp
RUN g++ -O2 -o /usr/local/bin/train_cpp /tmp/train.cpp && rm /tmp/train.cpp

# Install Python dependencies
COPY environment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
