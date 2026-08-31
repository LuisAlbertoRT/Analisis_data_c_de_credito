import time
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib

# -----------------------------------------------------------------------------
# 1. Configuración de Logging y App
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("fraud_api")

app = FastAPI(
    title="Fraud Detection API",
    description="Microservicio de inferencia en tiempo real para detección de fraude transaccional.",
    version="1.0.0"
)

# -----------------------------------------------------------------------------
# 2. Definición de la Clase Contenedora (Production Wrapper)
# -----------------------------------------------------------------------------
class ProductionFraudClassifier:
    """
    Wrapper que encapsula el modelo XGBoost entrenado y aplica el umbral óptimo
    calculado out-of-fold durante el proceso de validación.
    """
    def __init__(self, model_path: str, threshold: float = 0.8878):
        self.threshold = threshold
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Modelo cargado exitosamente desde {model_path}")
        except Exception as e:
            logger.error(f"Error al cargar el artefacto del modelo: {e}")
            raise e

    def predict_prob(self, df: pd.DataFrame) -> np.ndarray:
        # Devuelve la probabilidad continua p(y=1)
        return self.model.predict_proba(df)[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # Devuelve la clase binaria basada en el umbral óptimo
        probs = self.predict_prob(df)
        return (probs >= self.threshold).astype(int)

# Cargar el modelo al iniciar la API
MODEL_PATH = "Salidas/Modelos_finales/modelo_fraude_xgboost_final.joblib"
OPTIMAL_THRESHOLD = 0.8878

try:
    classifier = ProductionFraudClassifier(model_path=MODEL_PATH, threshold=OPTIMAL_THRESHOLD)
except Exception:
    # Si el archivo del modelo no existe en local, se inicializa como None para permitir el arranque de desarrollo
    classifier = None
    logger.warning(f"No se pudo cargar el archivo en '{MODEL_PATH}'. El endpoint de inferencia fallará hasta que el artefacto esté disponible.")


# -----------------------------------------------------------------------------
# 3. Esquemas de Datos (Pydantic Models)
# -----------------------------------------------------------------------------
class TransactionInput(BaseModel):
    transaction_id: str = Field(..., example="tx_9876543210")
    timestamp: float = Field(..., description="Marca de tiempo en segundos", example=147980.0)
    amount: float = Field(..., ge=0.0, description="Monto de la transacción", example=149.99)
    # Lista de variables numéricas anónimas V1 a V28
    v_features: List[float] = Field(
        ..., 
        min_items=28, 
        max_items=28, 
        description="Vector de 28 variables numéricas anónimas (V1 a V28)",
        example=[-1.3598, -0.0727, 2.5363, 1.3781, -0.3383, 0.4623, 0.2395, 0.0986, 0.3637, 0.0907, -0.5516, -0.6178, -0.9913, -0.3111, 1.4681, -0.4704, 0.2079, 0.0257, 0.4039, 0.2514, -0.0183, 0.2778, -0.1104, 0.0669, 0.1285, -0.1891, 0.1335, -0.0210]
    )

class BatchTransactionInput(BaseModel):
    transactions: List[TransactionInput]

class PredictionResult(BaseModel):
    transaction_id: str
    fraud_score: float = Field(..., description="Probabilidad predicha de fraude [0.0 - 1.0]")
    is_fraud: bool = Field(..., description="Decisión binaria con umbral 0.8878")
    action: str = Field(..., description="Acción sugerida: APPROVE o REJECT_OR_REVIEW")
    latency_ms: float = Field(..., description="Tiempo de inferencia en milisegundos")

class BatchPredictionResponse(BaseModel):
    total_processed: int
    frauds_detected: int
    results: List[PredictionResult]

# -----------------------------------------------------------------------------
# 4. Pipeline de Preprocesamiento en Tiempo Real
# -----------------------------------------------------------------------------
def preprocess_transaction(tx: TransactionInput) -> pd.DataFrame:
    """
    Transforma la entrada JSON al formato exacto de variables que espera el modelo.
    """
    # 1. Extracción de variables de la hora y cíclicas
    hour = (tx.timestamp / 3600.0) % 24
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    is_night_transaction = 1 if 0.0 <= hour < 6.0 else 0

    # 2. Transformaciones del monto
    amount_log = np.log1p(tx.amount)
    es_cero = 1 if tx.amount == 0 else 0
    es_uno_o_menos = 1 if tx.amount <= 1.0 else 0

    # 3. Construcción del diccionario de características (V1 a V28 + Ingenierizadas)
    data = {}
    for i, val in enumerate(tx.v_features, start=1):
        data[f"V{i}"] = val

    data["amount_log"] = amount_log
    data["es_cero"] = es_cero
    data["es_uno_o_menos"] = es_uno_o_menos
    data["hour_sin"] = hour_sin
    data["hour_cos"] = hour_cos
    data["is_night_transaction"] = is_night_transaction

    return pd.DataFrame([data])

# -----------------------------------------------------------------------------
# 5. Endpoints de la API
# -----------------------------------------------------------------------------
@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Endpoint de monitoreo de estado para Kubernetes / Load Balancer."""
    return {
        "status": "online",
        "model_loaded": classifier is not None,
        "threshold": OPTIMAL_THRESHOLD
    }

@app.post("/predict", response_model=PredictionResult, status_code=status.HTTP_200_OK)
def predict_single_transaction(transaction: TransactionInput):
    """
    Evalúa una única transacción en tiempo real.
    """
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="El modelo de evaluación no está disponible."
        )

    start_time = time.time()

    # Preprocesamiento de datos
    df_features = preprocess_transaction(transaction)

    # Inferencia
    try:
        score = float(classifier.predict_prob(df_features)[0])
        is_fraud_bool = bool(score >= classifier.threshold)
        action = "REJECT_OR_REVIEW" if is_fraud_bool else "APPROVE"
    except Exception as e:
        logger.error(f"Error durante la inferencia para {transaction.transaction_id}: {e}")
        raise HTTPException(status_code=500, detail="Error interno al ejecutar la inferencia.")

    latency = round((time.time() - start_time) * 1000, 2)

    logger.info(f"TxID: {transaction.transaction_id} | Score: {score:.4f} | Decision: {action} | Latencia: {latency}ms")

    return PredictionResult(
        transaction_id=transaction.transaction_id,
        fraud_score=round(score, 6),
        is_fraud=is_fraud_bool,
        action=action,
        latency_ms=latency
    )

@app.post("/predict/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK)
def predict_batch_transactions(batch: BatchTransactionInput):
    """
    Evalúa un lote de transacciones de forma vectorizada.
    """
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="El modelo de evaluación no está disponible."
        )

    start_time = time.time()
    
    # Preprocesamiento por lotes
    frames = [preprocess_transaction(tx) for tx in batch.transactions]
    df_batch = pd.concat(frames, ignore_index=True)

    # Inferencia vectorizada
    scores = classifier.predict_prob(df_batch)
    
    results = []
    frauds_count = 0

    for idx, tx in enumerate(batch.transactions):
        score = float(scores[idx])
        is_fraud = bool(score >= classifier.threshold)
        if is_fraud:
            frauds_count += 1
            
        results.append(PredictionResult(
            transaction_id=tx.transaction_id,
            fraud_score=round(score, 6),
            is_fraud=is_fraud,
            action="REJECT_OR_REVIEW" if is_fraud else "APPROVE",
            latency_ms=0.0  # Calculado a nivel de lote
        ))

    total_latency = round((time.time() - start_time) * 1000, 2)
    logger.info(f"Lote procesado: {len(batch.transactions)} transacciones | Fraudes: {frauds_count} | Latencia total: {total_latency}ms")

    return BatchPredictionResponse(
        total_processed=len(results),
        frauds_detected=frauds_count,
        results=results
    )