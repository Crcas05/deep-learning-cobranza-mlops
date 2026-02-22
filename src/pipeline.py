"""
Pipeline principal de Batch Prediction
Ejecuta: Bronze → Silver → Gold → Model → Batch Predictions
"""

import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.generate_data import generate_bronze_data
from src.preprocessing import process_silver
from src.gold_dataset import create_gold_dataset
from src.train_model import train_model
from src.predict_batch import batch_predict


def main():
    logger.info("🚀 Iniciando pipeline MLOps...")

    logger.info("📥 Etapa BRONZE...")
    generate_bronze_data()

    logger.info("🔄 Etapa SILVER...")
    process_silver()

    logger.info("✨ Etapa GOLD...")
    create_gold_dataset()

    model_path = Path("models/local/model.keras")

    if not model_path.exists():
        logger.info("🤖 Entrenando modelo...")
        train_model()
    else:
        logger.info("✅ Modelo existe, saltando entrenamiento")

    logger.info("🔮 Etapa BATCH PREDICTION...")
    batch_predict()

    logger.info("✅ Pipeline completado exitosamente!")


if __name__ == "__main__":
    main()