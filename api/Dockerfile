# Use an NVIDIA GPU-enabled base image
FROM nvidia/cuda:11.0.3-base-ubuntu20.04
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip
#     && rm -rf /var/lib/apt/lists/*
# RUN python3 -m ensurepip --default-pip


# Relevant folder
ARG FOLDER=/api

# Create a folder
RUN mkdir -p $FOLDER

# Install packages
COPY ./requirements.txt $FOLDER/requirements.txt
RUN pip3 install -r $FOLDER/requirements.txt
RUN pip3 install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/  # Use cu117 if on CUDA 11.7
# Copy the project files into the container
COPY ./src $FOLDER/src

# Expose any necessary ports
EXPOSE 7860

# Set the working directory
WORKDIR $FOLDER/src

# Start the application
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "7860", "--reload", "--reload-dir", "/api", "main:app"]