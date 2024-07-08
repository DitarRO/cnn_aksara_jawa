FROM python:3.9.19-slim-bullseye
WORKDIR /app
RUN pip install tensorflow==2.15 flask pillow
COPY . .
CMD ["python", "app.py"]
