FROM python:3.9-slim

WORKDIR /app/backend/disease_prediction

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip3 install -U scikit-learn

COPY . .

RUN pip3 install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
