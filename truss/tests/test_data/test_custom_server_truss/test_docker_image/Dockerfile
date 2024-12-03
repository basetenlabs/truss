FROM python:3.11-slim

# Update package lists and install curl
RUN apt-get update && apt-get install -y curl

# Install pip dependencies
RUN pip install fastapi[standard]

# Copy the FastAPI application code
COPY app.py /home/app.py
COPY main.py /home/main.py

# Set the working directory
WORKDIR /home

# Command to run FastAPI directly
ENTRYPOINT ["python", "main.py"]
