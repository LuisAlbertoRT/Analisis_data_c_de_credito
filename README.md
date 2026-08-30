# 🛡️ Fraud Detector XGBoost

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-v2.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-green.svg)]()

Sistema de clasificación supervisada basado en **XGBoost** diseñado para la detección en tiempo real de transacciones financieras fraudulentas bajo entornos de **alto desbalance de clases**.

---

## 📌 Descripción del Proyecto

El proyecto implementa un pipeline completo de Machine Learning orientado a resolver el problema de fraude transaccional. La solución prioriza la métrica **PR-AUC** y el ajuste fino del umbral de decisión mediante la optimización de **$F_2\text{-Score}$**, reduciendo al mínimo los Falsos Positivos mientras mantiene una alta cobertura (Recall) de transacciones sospechosas.

### 🔑 Aspectos Clave
* **Tratamiento del Desbalance:** Optimización mediante `scale_pos_weight` (~201.21) y función de pérdida adaptada a curvas Precision-Recall (`aucpr`).
* **Umbral de Decisión Óptimo:** Calibrado en **$\tau = 0.9760$** para maximizar el retorno de negocio y minimizar la fricción con clientes legítimos.
* **Control de Overfitting:** Regularización estricta (`max_depth=5`, `subsample=0.965`, `colsample_bytree=0.626`).

---

## 📊 Resultados y Desempeño en Prueba (Test Set)

Evaluado sobre un conjunto de prueba de **56,962 transacciones**:

| Métrica | Valor Obtenido | Descripción |
| :--- | :---: | :--- |
| **PR-AUC (Principal)** | **`0.8738`** | Área bajo la curva Precision-Recall |
| **ROC-AUC (Secundaria)**| **`0.9859`** | Área bajo la curva ROC |
| **Precision** | **`96.34%`** | De cada 100 alertas, ~96 son fraudes reales |
| **Recall** | **`80.61%`** | Captura del 80.61% del fraude total |
| **$F_2\text{-Score}$** | **`0.8333`** | Balance con mayor peso en la captura de fraude |
| **FPR (Falsas Alarmas)**| **`0.0053%`** | Solo 3 falsos positivos en 56,864 casos legítimos |

### 🧮 Matriz de Confusión ($\tau = 0.9760$)

```text
                  Predicción Legítimo (0)    Predicción Fraude (1)
Real Legítimo (0)          56,861 (TN)                 3 (FP)
Real Fraude (1)                19 (FN)                79 (TP)



.
├── Archivos_aux/
│   ├── raw/                  # Datasets originales
│   └── processed/            # Datasets transformados
├── Notebooks/
│   ├── model_metadata.json   # Metadatos del modelo e hiperparámetros
│   └── model.joblib          # Binario serializado del modelo final
├── Salidas/
│   └── predicciones_test.csv # Predicciones exportadas (id_solicitante, score_modelo)
├── MODEL_CARD.md             # Documentación detallada del modelo
├── README.md                 # Visión general del proyecto
└── requirements.txt          # Dependencias de Python