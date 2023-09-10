# Use the official Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the FastAPI application files to the container
COPY ./app /app

# Copy requirements.txt, main.py, and PRODUCTS.csv to the container
COPY requirements.txt main.py PRODUCTS.csv /app/


# Create a virtual environment and install application dependencies
RUN python3 -m venv venv
RUN /app/venv/bin/pip install --no-cache-dir -r ./requirements.txt

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Run the FastAPI application
CMD ["/app/venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
