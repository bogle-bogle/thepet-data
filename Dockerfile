# Use the official Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app


# Create and activate a virtual environment
RUN python3 -m venv venv
RUN . venv/bin/activate

# Copy the FastAPI application files to the container
COPY ./app /app

# Copy requirements.txt to the container
COPY requirements.txt /app

# Copy main.py and PRODUCTS.csv to the container
COPY main.py /app
COPY PRODUCTS.csv /app

# Install application dependencies
RUN pip install --no-cache-dir -r ./requirements.txt

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
