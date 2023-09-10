FROM python:3.9-slim

WORKDIR /app/backend/product-recommendation

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip3 install -r requirements.txt

EXPOSE 8006

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8006"]