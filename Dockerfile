FROM python:3.12-slim

WORKDIR /app

# Install only runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy web app + config + model
COPY src/config.py src/config.py
COPY src/logger.py src/logger.py
COPY web/ web/
COPY models/demand_model.pkl models/demand_model.pkl
COPY models/model_meta.pkl models/model_meta.pkl
COPY data/processed/preprocessed_data_3month_enriched.csv data/processed/preprocessed_data_3month_enriched.csv

EXPOSE 5000

CMD ["python", "web/app.py"]
