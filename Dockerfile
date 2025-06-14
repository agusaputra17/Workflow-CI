# Gunakan base image dengan Python
FROM python:3.12.7

# Buat working directory
WORKDIR /app

# Copy semua file dari folder lokal ke dalam container
COPY . .

# Install dependensi
RUN pip install --no-cache-dir --upgrade pip && \
    pip install \
    mlflow==2.19.0 \
    cloudpickle==3.1.1 \
    numpy==2.0.2 \
    tensorflow==2.18.0

# Set environment variable untuk MLflow Tracking URI
ENV MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# Jalankan skrip Python (sesuai kebutuhan)
CMD ["python", "MLproject/modelling.py"]