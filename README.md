# 🛡️ Fraud Detection Engine — XGBoost Architecture

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?style=flat-square)
![PR-AUC](https://img.shields.io/badge/PR--AUC-0.8860-brightgreen?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production--Ready-success?style=flat-square)

Solución de Machine Learning para la detección y mitigación de fraude transaccional en entornos de alto desbalance de clases. El sistema utiliza una arquitectura basada en **XGBoost Classifier** optimizado mediante binarización de probabilidad y análisis fuera de muestra.


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
│   └── datos_fraud.csv                 
├── Salidas/Modelos_finales
│   ├── EDA-datos_fraud.html           
│   ├── predicciones_test.csv           
│   └── model_metadata.json             
├── Noteboooks/
│   └── Proceso bajo el que se fue construyendo el modelo   #Sirven de auxiliares
├── D Resolucion.ipynb       
├── MODEL_CARD.md                       
└── README.md                           
```

```mermaid
flowchart TD
    A[Conjunto de Datos Original: datos_fraud.csv] --> B[Deduplicación y Limpieza de Datos]
    B --> C[Ingeniería y Transformación de Variables]
    
    subgraph Feature_Engineering [Ingeniería de Variables]
        C1[Transformación Logarítmica: amount_log con np.log1p]
        C2[Indicadores Binarios: es_cero, es_uno_o_menos]
        C3[Transformaciones Cíclicas: hour_sin / hour_cos]
        C4[Flag Horario Nocturno: is_night_transaction]
        C5[Deciles Fuera de Muestra - Out-of-Fold]
    end
    
    C --> Feature_Engineering
    Feature_Engineering --> D[División Estratificada Entrenar / Probar 80/20]
    
    D --> E[Evaluación Comparativa de Modelos Iniciales]
    E -->|Búsqueda de Hiperparámetros con Optuna| F[Métrica Objetivo de Optimización: PR-AUC]
    
    F --> G[Clasificador XGBoost + Scale Pos Weight = 201.21]
    G --> H[Cálculo del Umbral Óptimo Out-of-Fold en Entrenar: 0.9474]
    H --> I[Encapsulamiento en Envoltorio Producción: ProductionFraudClassifier]
    I --> J[Generación de Artefactos .joblib, model_metadata.json y predicciones_test.csv]

    ```