# Use an official Python runtime as a parent image
FROM python:3.7

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
ADD . /usr/src/app

# Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y nano virtualenv

# Run setup_virtualenv.sh when the container launches
CMD ["./setup_virtualenv.sh"]
