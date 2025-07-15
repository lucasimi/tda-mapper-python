FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install .[app]

CMD ["tda-mapper-app"]