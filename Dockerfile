FROM tensorflow/tensorflow:2.11.0-gpu

WORKDIR /app
RUN mkdir saved

COPY *.py .
COPY static/ .
COPY templates/ .
COPY requirements.txt .

RUN /usr/bin/python3 -m pip install --upgrade pip

RUN pip install -r requirements.txt

CMD ["python", "main.py"]
