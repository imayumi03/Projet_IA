FROM python:3.10-slim

WORKDIR /app

COPY app.py model.py lstm_trained.pth requirements.txt ./

RUN pip install --no-deps --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
