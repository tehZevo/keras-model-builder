FROM python:3.10

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

VOLUME ["/app/models"]

COPY . .

ENTRYPOINT [ "python", "-u", "main.py" ]
