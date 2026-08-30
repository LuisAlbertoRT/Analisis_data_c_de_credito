# Model Card: Fraud_Detector_XGBoost v1.0.0

## 1. Descripción General del Modelo
* **Nombre del Modelo:** `Fraud_Detector_XGBoost`
* **Versión:** `1.0.0`
* **Fecha de Creación:** 30 de agosto de 2026
* **Autor / Propietario:** Equipo de Data Science / Analytics
* **Tipo de Modelo:** Clasificación Binaria Supervisada
* **Framework:** XGBoost (`XGBClassifier`) / Scikit-Learn
* **Propósito:** Detección en tiempo real de transacciones sospechosas de fraude con un alto grado de desbalance de clases.

---

## 2. Uso Intencionado y Limitaciones
* **Uso Primario:** Evaluación de riesgo de transacciones financieras para emitir una alerta de bloqueo preventivo o revisión manual.
* **Casos Fuera de Alcance:** No debe utilizarse como único criterio para la cancelación permanente de cuentas de usuario sin validación de un analista humano.
* **Consideraciones del Dominio:** El modelo fue diseñado considerando una tasa alta de desbalance ($scale\_pos\_weight \approx 201.21$). Si la distribución de la tasa base de fraude en producción cambia significativamente, se requiere un reentrenamiento o recalibración del umbral.

---

## 3. Esquema de Datos (Data Schema)
* **Variable Objetivo (Target):** `es_fraude` ($0 = \text{Legítimo}$, $1 = \text{Fraude}$)
* **Número Total de Features:** 41 variables de entrada

### Variables Predictoras (Input Features)
* **Variables Anonimizadas (32):** `variable_01` a `variable_32`
* **Transformaciones Financieras (3):** `amount_log`, `es_uno_o_menos`, `es_cero`
* **Variables Temporales y Cíclicas (5):** `hour`, `day`, `is_night_transaction`, `hour_sin`, `hour_cos`
* **Segmentación Operativa (1):** `deciles`

---

## 4. Configuración del Modelo e Hiperparámetros
* **Objetivo de Optimización:** `binary:logistic`
* **Métrica de Evaluación Interna:** `aucpr` (Area Under the Precision-Recall Curve)
* **Control de Overfitting:**
  * `max_depth`: 5
  * `learning_rate`: 0.0516
  * `colsample_bytree`: 0.6261
  * `subsample`: 0.9658
  * `early_stopping_rounds`: 20
* **Ajuste de Desbalance de Clase:** `scale_pos_weight` = 201.209

---

## 5. Estrategia de Inferencia y Umbral de Decisión
Para priorizar el retorno del negocio y balancear el costo de perder fraudes vs. la fricción con clientes legítimos, se optimizó el **$F_2\text{-Score}$** sobre la curva Precision-Recall.

* **Umbral Óptimo de Decisión ($\tau$):** `0.9760`
* **Regla de Inferencia en Producción:**
  $$\hat{y} = \begin{cases} 1 (\text{Alerta de Fraude}), & \text{si } P(\text{fraude} \mid X) \ge 0.9760 \\ 0 (\text{Transacción Legítima}), & \text{si } P(\text{fraude} \mid X) < 0.9760 \end{cases}$$

---

## 6. Rendimiento y Métricas de Evaluación (Set de Prueba / Test)

### Métricas Globales
| Métrica | Resultado en Test | Brecha de Generalización (Train - Test) |
| :--- | :--- | :--- |
| **PR-AUC (Métrica Principal)** | **0.8738** | 0.0899 |
| **ROC-AUC (Métrica Secundaria)** | **0.9859** | 0.0140 |

### Métricas Operativas en el Umbral $\tau = 0.9760$
* **Precisión (Precision):** $96.34\%$ (De cada 100 alertas emitidas, ~96 son fraudes reales)
* **Cobertura (Recall):** $80.61\%$ (Captura el $80.61\%$ de todos los fraudes del dataset)
* **$F_1\text{-Score}$:** 0.8778
* **$F_2\text{-Score}$:** 0.8333
* **Tasa de Falsas Alarmas (FPR):** $0.0053\%$ (Solo 3 falsos positivos de 56,864 transacciones legítimas)

### Matriz de Confusión en Test ($N = 56,962$)
| | Predicción Legítimo ($0$) | Predicción Fraude ($1$) | Total Real |
| :--- | :--- | :--- | :--- |
| **Real Legítimo ($0$)** | **56,861** (TN) | **3** (FP) | 56,864 |
| **Real Fraude ($1$)** | **19** (FN) | **79** (TP) | 98 |

---

## 7. Importancia de Variables (Top 10 Feature Importance)
La importancia fue calculada con base en el peso del ganancia (*Gain*) del árbol de XGBoost:

| Ranking | Variable | Importancia Relativa (%) |
| :---: | :--- | :---: |
| 1 | `variable_15` | **31.91%** |
| 2 | `variable_19` | **10.25%** |
| 3 | `variable_17` | **5.50%** |
| 4 | `variable_25` | **4.06%** |
| 5 | `deciles` | **2.95%** |
| 6 | `variable_12` | **2.42%** |
| 7 | `variable_16` | **2.07%** |
| 8 | `variable_09` | **2.02%** |
| 9 | `variable_08` | **1.84%** |
| 10 | `variable_10` | **1.83%** |

---

## 8. Consideraciones Operativas y Mantenimiento
1. **Monitoreo de Data Drift:** Realizar seguimiento mensual a la distribución de `variable_15`, `variable_19` y `variable_17`, ya que acumulan casi el $50\%$ de la capacidad explicativa del modelo.
2. **Re-evaluación del Umbral:** Si las políticas del negocio deciden reducir los falsos negativos a costa de aceptar más revisiones manuales, ajustar el umbral por debajo de `0.9760`.
3. **Latencia de Inferencia:** Optimizado para ejecutarse en pipelines en tiempo real (formato de entrada en array NumPy/Pandas de 41 columnas).