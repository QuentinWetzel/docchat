FROM python:3.11-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 7860
# Railway provides $PORT; app.py falls back to 7860 locally.
CMD ["python", "app.py"]
