# Model Card: Fraud_Detector_XGBoost

## 1. Información General
* **Nombre del Modelo:** `Fraud_Detector_XGBoost`
* **Versión:** `1.0.0`
* **Fecha de Creación:** 30 de agosto de 2026
* **Autor:** Luis Alberto Rueda Tapia
* **Framework:** XGBoost (`XGBClassifier`) / Scikit-Learn
* **Descripción:** Modelo de aprendizaje supervisado diseñado para la detección de fraude transaccional en conjuntos de datos altamente desbalanceados.

---

## 2. Uso Intencionado y Limitaciones
* **Caso de Uso Principal:** Clasificación transaccional en tiempo real o en lote para predecir si una operación es fraudulenta (`is_fraud = 1`) o legítima (`is_fraud = 0`).
* **Criterio de Decisión:**
  * **Umbral Óptimo de Decisión:** `0.9812`
  * **Criterio de Optimización:** Maximización de F1-Score y curva Precision-Recall.

---

## 3. Esquema de Datos y Variables
* **Variable Objetivo:** `is_fraud`
* **Número de Features:** 41 variables de entrada
* **Features Destacadas:**
  * Variables numéricas / transformadas (`variable_01` a `variable_32`)
  * Transformaciones temporales y de monto (`amount_log`, `hour`, `day`, `is_night_transaction`, `hour_sin`, `hour_cos`)
  * Bins y discretización (`deciles`, `es_uno_o_menos`, `es_cero`)

---

## 4. Métricas de Rendimiento (Test Set)

### Evaluación Cuantitativa
| Métrica | Valor | Descripción / Observaciones |
| :--- | :--- | :--- |
| **PR-AUC** | `0.8738` | Métrica principal dada la naturaleza desbalanceada |
| **ROC-AUC** | `0.9859` | Capacidad global de discriminación del modelo |
| **Precision** | `0.9630` | El 96.3% de las alertas generadas son fraude real |
| **Recall** | `0.7959` | El modelo captura el 79.6% del fraude total |
| **F1-Score** | `0.8715` | Balance armónico entre Precisión y Recall |
| **F2-Score** | `0.8245` | Ponderación priorizando la captura de fraudes |
| **FPR** | `0.000053` | Tasa extremadamente baja de falsos positivos (0.005%) |

### Matriz de Confusión en Test
```text
                  Predicho Legítimo    Predicho Fraude
Real Legítimo        56,861 (TN)           3 (FP)
Real Fraude              20 (FN)          78 (TP)
```
* **Verdaderos Negativos (TN):** 56,861 transacciones legítimas correctamente identificadas.
* **Falsos Positivos (FP):** Solo 3 falsas alarmas enviadas a revisión.
* **Falsos Negativos (FN):** 20 casos de fraude no detectados.
* **Verdaderos Positivos (TP):** 78 fraudes interceptados correctamente.

### Brecha de Generalización (Gap Train-Test)
* **PR-AUC Gap:** `0.0899`
* **ROC-AUC Gap:** `0.0140`

---

## 5. Importancia de Variables (Top 10 Features)

| Posición | Variable | Weight / Gain Ratio |
| :---: | :--- | :---: |
| 1 | `variable_15` | `0.3191` |
| 2 | `variable_19` | `0.1025` |
| 3 | `variable_17` | `0.0550` |
| 4 | `variable_25` | `0.0406` |
| 5 | `deciles` | `0.0295` |
| 6 | `variable_12` | `0.0242` |
| 7 | `variable_16` | `0.0207` |
| 8 | `variable_09` | `0.0202` |
| 9 | `variable_08` | `0.0184` |
| 10 | `variable_10` | `0.0183` |

---

## 6. Hiperparámetros Clave del Entrenador
* **`n_estimators`:** `1500`
* **`learning_rate`:** `0.0516`
* **`max_depth`:** `5`
* **`scale_pos_weight`:** `201.21` *(Ajuste clave para balancear la clase minoritaria)*
* **`subsample`:** `0.9658`
* **`colsample_bytree`:** `0.6261`
* **`eval_metric`:** `aucpr`
* **`early_stopping_rounds`:** `20`

---

## 7. Consideraciones de Monitoreo y Mantenimiento
1. **Data Drift:** Monitorear periódicamente la distribución de las variables top (`variable_15`, `variable_19`, `variable_17`) mediante KS-test o PSI (Population Stability Index).
2. **Concept Drift:** Monitorear semanalmente la tasa de Falsos Negativos y el PR-AUC real conforme se obtengan las confirmaciones o reclamos por fraude.
3. **Reentrenamiento:** Se recomienda reentrenar el modelo si el PR-AUC cae por debajo de `0.80` o si el drift de las características principales supera un PSI de `0.25`.