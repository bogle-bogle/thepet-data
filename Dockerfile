# Use the official Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /code

# Copy requirements.txt, main.py, and PRODUCTS.csv to the container
COPY ./requirements.txt  /code/requirements.txt
COPY ./main.py  /code/app/main.py
COPY ./PRODUCTS.csv  /code/app/PRODUCTS.csv
COPY ./config.py  /code/app/config.py

# Create a virtual environment and install application dependencies
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy the FastAPI application files to the container
COPY ./app /code/app

# Run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
