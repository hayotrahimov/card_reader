FROM python:3
COPY server.py /
COPY requirements.txt /
RUN apt-get install libsm6 libxext6  -y
RUN pip install -r requirements.txt
CMD [ "python", "-u", "server.py" ]
