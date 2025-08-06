# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY ./requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY ./app ./app

# Expose port 8000 to allow communication to the uvicorn server
EXPOSE 8000

# Run main.py when the container launches
# Use --host 0.0.0.0 to make the server accessible from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]