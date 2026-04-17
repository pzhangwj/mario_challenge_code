# Use an official Python runtime as a parent image
FROM python:3.8.15

# Vérifier la version installée
RUN python --version

# Ajoutez ceci à votre Dockerfile pour installer libGL
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Mettre à jour pip, setuptools et wheel
RUN pip install --upgrade pip setuptools wheel

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Make ports available to the world outside this container
EXPOSE 8080

RUN mkdir -p /app/output

# Set environment variable
ENV Team_name=DF41  

# Default command to run inference_task1.py
CMD ["python", "inference_task1.py"]
