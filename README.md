# 🛡️ Fraud Detection Engine — XGBoost Architecture

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?style=flat-square)
![PR-AUC](https://img.shields.io/badge/PR--AUC-0.8860-brightgreen?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production--Ready-success?style=flat-square)

Solución de Machine Learning de extremo a extremo para la detección y mitigación de fraude transaccional en entornos de alto desbalance de clases. El sistema utiliza una arquitectura basada en **XGBoost Classifier** optimizado mediante binarización de probabilidad y análisis fuera de muestra (Out-of-Fold).


## 🎯 Descripción del Problema

El objetivo es clasificar y prevenir operaciones fraudulentas en solicitudes transaccionales mediante datos históricos. 

### Retos Clave:
* **Desbalance Severo:** El conjunto de datos cuenta con menos del **0.2%** de casos positivos de fraude (492 transacciones de 284,807).
* **Anonimización:** Variables numéricas transformadas donde solo se dispone directamente del sello de tiempo, monto y etiqueta.
* **Costo Asimétrico:** Alto costo de los Falsos Negativos (fraude no capturado) frente al costo por fricción operativa de Falsos Positivos (bloqueo legítimo).

---

## 📁 Estructura del Repositorio

```text
.
├── Archivos_aux/
│   └── datos_fraud.csv                 # Dataset original transaccional
├── Salidas/
│   ├── EDA-datos_fraud.html            # Reporte interactivo de análisis exploratorio (Sweetviz)
│   ├── predicciones_test.csv           # Predicciones finales (ID Solicitante + Probability Score)
│   ├── modelo_fraude_xgboost_final.joblib # Artefacto empaquetado (Production Wrapper + Model)
│   └── model_metadata.json             # Metadatos, esquema de datos e hiperparámetros
├── Documentos/
│   └── Explicacion_Punto_Corte.pdf     # Justificación técnica del umbral de decisión operativo
├── Notebook_Fraude_XGBoost.ipynb       # Jupyter Notebook documentado de end-to-end (EDA a Testing)
├── MODEL_CARD.md                       # Documentación formal de la tarjeta del modelo
└── README.md                           # Documentación principal del proyecto
```

```mermaid
flowchart TD
    A[Raw Dataset: datos_fraud.csv] --> B[Deduplicación & Limpieza Limpia]
    B --> C[Feature Engineering & Transformación]
    
    subgraph Feature_Engineering [Ingeniería de Variables]
        C1[amount_log: np.log1p]
        C2[Flags: es_cero, es_uno_o_menos]
        C3[Ciclos Cíclicos: hour_sin / hour_cos]
        C4[Horario Nocturno: is_night_transaction]
        C5[Deciles Out-of-Fold]
    end
    
    C --> Feature_Engineering
    Feature_Engineering --> D[Stratified Train / Test Split 80/20]
    
    D --> E[Benchmark de Modelos Iniciales]
    E -->|Optuna Hyperparameter Search| F[Optimización Objetivo: PR-AUC Target]
    
    F --> G[XGBoost Classifier + Scale Pos Weight = 201.21]
    G --> H[Cálculo de Umbral Óptimo OOF en Train: 0.9474]
    H --> I[Empaquetamiento en Wrapper ProductionFraudClassifier]
    I --> J[Generación de Artefactos .joblib, model_metadata.json y predicciones_test.csv]

    ```