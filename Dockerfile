# Use the official Python base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the FastAPI application files to the container
COPY ./app /app

# Install application dependencies
RUN pip install --no-cache-dir -r ./requirements.txt

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
