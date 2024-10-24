FROM python:3.11-slim

# Update package lists and install curl
RUN apt-get update && apt-get install -y curl

# Install FastAPI
RUN pip install fastapi[standard]

# Copy the FastAPI application code
COPY app.py /home/app.py

# Command to run FastAPI directly
ENTRYPOINT ["fastapi", "dev", "/home/app.py"]
