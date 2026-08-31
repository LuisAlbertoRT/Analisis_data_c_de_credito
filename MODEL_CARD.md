# 📋 Model Card: Fraud_Detector_XGBoost

Documentación técnica detallada y ficha de desempeño para el modelo supervisado de prevención y detección de fraude transaccional en producción[cite: 5].

---

##  1. Visión General del Modelo

* **Nombre del Modelo:** `Fraud_Detector_XGBoost`[cite: 5]
* **Versión:** `1.0.0`[cite: 5]
* **Tipo de Algoritmo:** Gradient Boosted Decision Trees (`XGBClassifier`)[cite: 5]
* **Framework / Entorno:** `xgboost` (v1.7+) / `scikit-learn` / Python 3.10+[cite: 5]
* **Fecha de Entrenamiento:** 2026-08-30 18:57:48[cite: 5]
* **Autor / Propietario:** Equipo de Data Science / Analytics[cite: 5]
* **Descripción Resumida:** Modelo de clasificación binaria optimizado para entornos de extremo desbalance de clases (prevalencia de fraude < 0.2%). Evalúa solicitudes transaccionales para clasificarlas como aprobadas (`0`) o bloqueadas por riesgo de fraude (`1`)[cite: 5].

---

##  2. Uso Intencionado & Alcance

### Usos Intencionados (In-Scope)
* Clasificación en tiempo real o en lote (*batch*) de solicitudes transaccionales financieras.
* Priorización de alertas en motores de riesgo transaccional para revisión manual o bloqueo automático.

### Usos No Intencionados (Out-of-Scope)
* Evaluación de riesgo crediticio para otorgamiento de préstamos a largo plazo.
* Modelado de comportamiento de clientes fuera del contexto de transacciones financieras individuales.
* Inferencia sobre variables no estandarizadas o sin el pipeline previo de ingeniería de características.

---

##  3. Hiperparámetros de Producción

Configuración óptima del estimador `XGBClassifier` seleccionada mediante optimización bayesiana enfocada en la métrica PR-AUC[cite: 5]:

```json
{
  "objective": "binary:logistic",
  "eval_metric": "aucpr",
  "learning_rate": 0.038919997563362715,
  "n_estimators": 1500,
  "early_stopping_rounds": 30,
  "max_depth": 6,
  "subsample": 0.8001201242229923,
  "colsample_bytree": 0.710799660783632,
  "scale_pos_weight": 223.94350677075371,
  "reg_alpha": 2.006923372947634,
  "reg_lambda": 3.477362954209379,
  "gamma": 3.4232564999517594,
  "enable_categorical": true,
  "random_state": 42,
  "n_jobs": -1
}
```[cite: 5]

---

##  4. Umbral de Decisión Operativo (Thresholding)

* **Umbral Óptimo de Corte (`Optimal Threshold`):** `0.506157`[cite: 5]
* **Criterio de Selección:** Maximización del balance mediante **F2-Score / Precision-Recall Optimization**[cite: 5]. Este criterio otorga el doble de importancia a la captura de fraudes (Recall) sobre la precisión, manteniendo a su vez la tasa de falsos positivos en niveles insignificantes para la operación[cite: 5].
* **Lógica de Inferencia:**
  * $\text{Score} \ge 0.506157 \implies \text{Bloquear / Fraude (1)}$[cite: 5]
  * $\text{Score} < 0.506157 \implies \text{Aprobar / Legítimo (0)}$[cite: 5]

---

##  5. Evaluación de Desempeño (Test Set Evaluation)

Evaluación imparcial realizada sobre un conjunto de prueba aislado de **56,746 observaciones**[cite: 5].

### Métricas Globales Discriminativas

| Métrica | Valor Obtenido | Descripción / Impacto Operativo |
| :--- | :---: | :--- |
| **PR-AUC (Métrica Primaria)** | **0.7968** | Área bajo la curva Precision-Recall. Evaluador clave en desbalance crítico[cite: 5]. |
| **ROC-AUC** | **0.9799** | Separabilidad probabilística global del modelo[cite: 5]. |
| **Precisión (Precision)** | **91.25%** | De cada 100 transacciones marcadas como fraude, 91.25 son fraudes reales[cite: 5]. |
| **Sensibilidad (Recall)** | **76.84%** | Cobertura directa del 76.84% del total de eventos fraudulentos existentes[cite: 5]. |
| **F1-Score** | **0.8343** | Media armónica balanceada entre Precisión y Sensibilidad[cite: 5]. |
| **F2-Score** | **0.7935** | Métrica ponderada con prioridad en la detección de fraudes (Recall)[cite: 5]. |
| **FPR (Tasa Falsos Positivos)**| **0.0124%** | Fricción mínima: solo ~1 de cada 8,000 transacciones legítimas se afecta[cite: 5]. |

### Matriz de Confusión en Test

```text
                        Predicción Legítimo (0)    Predicción Fraude (1)
Etiqueta Real Legítimo        56,644 (TN)                   7 (FP)
Etiqueta Real Fraude              22 (FN)                  73 (TP)
```[cite: 5]

### Brecha de Generalización (Generalization Gap)
* **Diferencia PR-AUC (Train - Test):** `0.0469` (Indica control efectivo contra el sobreajuste / *overfitting*)[cite: 5].
* **Diferencia ROC-AUC (Train - Test):** `0.0190`[cite: 5].

---

##  6. Importancia de Variables (Top 10 por Gain Weight)

Contribución relativa de las características al poder predictivo del algoritmo[cite: 5]:

| Ranking | Variable | Descripción Breve / Origen | Peso de Importancia (%) |
| :---: | :--- | :--- | :---: |
| **1** | `variable_15` | Característica latente anónima | **37.69%**[cite: 5] |
| **2** | `variable_19` | Característica latente anónima | **13.61%**[cite: 5] |
| **3** | `variable_17` | Característica latente anónima | **4.10%**[cite: 5] |
| **4** | `variable_25` | Característica latente anónima | **2.82%**[cite: 5] |
| **5** | `variable_21` | Característica latente anónima | **2.17%**[cite: 5] |
| **6** | `variable_12` | Característica latente anónima | **2.03%**[cite: 5] |
| **7** | `amount_log` | Monto transformado via $\ln(x+1)$ | **1.75%**[cite: 5] |
| **8** | `hour_cos` | Componente coseno de la hora cíclica | **1.66%**[cite: 5] |
| **9** | `variable_01` | Característica latente anónima | **1.58%**[cite: 5] |
| **10** | `variable_29` | Característica latente anónima | **1.48%**[cite: 5] |

---

##  7. Esquema de Datos de Entrada (41 Features)

El pipeline de entrada exige las siguientes 41 características procesadas en el orden especificado[cite: 5]:

```text
1.  variable_01          15. variable_15         29. variable_29         
2.  variable_02          16. variable_16         30. variable_30         
3.  variable_03          17. variable_17         31. variable_31         
4.  variable_04          18. variable_18         32. variable_32         
5.  variable_05          19. variable_19         33. amount_log          
6.  variable_06          20. variable_20         34. es_uno_o_menos      
7.  variable_07          21. variable_21         35. es_cero             
8.  variable_08          22. variable_22         36. hour                
9.  variable_09          23. variable_23         37. day                 
10. variable_10          24. variable_24         38. is_night_transaction
11. variable_11          25. variable_25         39. hour_sin            
12. variable_12          26. variable_26         40. hour_cos            
13. variable_13          27. variable_27         41. deciles             
14. variable_14          28. variable_28