# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.7-slim

ENV APP_HOME /app
WORKDIR $APP_HOME

# Install production dependencies.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r ./requirements.txt
RUN pip install tritonclient[all] --extra-index-url=https://pypi.ngc.nvidia.com  

# Copy local code to container image
COPY model.py  ./

CMD ["python", "client.py"]