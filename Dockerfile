FROM python:3.8
COPY server.py /
COPY requirements.txt /
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
CMD [ "python", "-u", "server.py" ]
