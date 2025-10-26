# Python 3.10 tabanlı imaj
FROM python:3.10-slim

# Çalışma dizini
WORKDIR /app

# Bağımlılıkları kopyala ve kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY . .

# Flask portu
EXPOSE 7860

# Flask uygulamasını çalıştır
CMD ["python", "cizim_api.py"]