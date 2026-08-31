# Model Card: Fraud Detector XGBoost (v1.0.0)

## 1. Visión General del Modelo
- **Nombre:** Fraud_Detector_XGBoost[cite: 7]
- **Versión:** 1.0.0[cite: 7]
- **Tipo de Modelo:** Clasiﬁcador Gradiente Gradiente Descentralizado (XGBoost / `XGBClassifier`)[cite: 7]
- **Fecha de creación:** 2026-08-30[cite: 7]
- **Desarrollado por:** Equipo de Data Science / Analytics[cite: 7]
- **Objetivo:** Predecir si una solicitud transaccional debe ser rechazada/bloqueada por riesgo de fraude (`es_fraude = 1`) o aprobada (`es_fraude = 0`)[cite: 6, 7].

---

## 2. Datos de Entrenamiento y Evaluación
- **Fuente de datos:** `datos_fraud.csv` (284,807 registros con 492 casos positivos de fraude, <0.2% de prevalencia)[cite: 6].
- **Esquema de División:** Split estratificado 80/20 manteniendo el balance de clase en entrenamiento y prueba[cite: 6].
- **Total de Características Utilizadas:** 41 variables (incluyendo 32 numéricas transformadas, indicadores de monto binarios y ciclos temporales)[cite: 6, 7].

---

## 3. Desempeño del Modelo

Métricas evaluadas sobre el **Conjunto de Test (In Sample Testing / Imparcial)** aplicando el **Umbral Óptimo de Decisión (0.9474)**[cite: 6, 7]:

### Métricas Globales Discriminativas
- **PR-AUC (Métrica Principal):** 0.8860[cite: 7]
- **ROC-AUC:** 0.9934[cite: 7]
- **Gap de Generalización PR-AUC (Train - Test):** 0.0921 (Generalización estable)[cite: 7]

### Matriz de Confusión en Test
- **Verdaderos Negativos (TN):** 56,646 legítimos aprobados correctamente[cite: 7]
- **Falsos Positivos (FP):** 5 legítimos bloqueados por error[cite: 7]
- **Falsos Negativos (FN):** 20 fraudes no detectados[cite: 7]
- **Verdaderos Positivos (TP):** 75 fraudes identificados y bloqueados[cite: 7]

### Métricas de Clasificación
- **Precision (Exactitud de la alerta):** 93.75%[cite: 7]
- **Recall (Cobertura de captura):** 78.95%[cite: 7]
- **F1-Score:** 0.8571[cite: 7]
- **F2-Score:** 0.8152[cite: 7]
- **Tasa de Falsos Positivos (FPR):** 0.0088%[cite: 7]

---

## 4. Importancia de Variables (Top 10 Gain Weights)
Las 10 variables con mayor peso en la discriminación de riesgo transaccional[cite: 7]:

1. `variable_15` (35.22%)[cite: 7]
2. `variable_19` (8.90%)[cite: 7]
3. `variable_17` (5.11%)[cite: 7]
4. `variable_25` (3.57%)[cite: 7]
5. `deciles` (2.94%)[cite: 7]
6. `amount_log` (2.37%)[cite: 7]
7. `variable_12` (2.36%)[cite: 7]
8. `variable_21` (2.34%)[cite: 7]
9. `day` (2.31%)[cite: 7]
10. `variable_16` (2.01%)[cite: 7]

---

## 5. Consideraciones de Despliegue y Ética
- **Manejo de Falsos Positivos:** Con una precisión del 93.75% y solo 5 falsos positivos detectados en la prueba, el impacto por fricción operacional o frustración de clientes legítimos es sumamente bajo[cite: 7].
- **Uso Recomendado:** Inferencia automatizada de riesgo para decisiones en línea (Aprobar/Bloquear o Derivar a revisión manual transacciones cercanas al umbral).
- **Limitaciones:** El modelo requiere que las variables de entrada sigan el esquema numérico estandarizado y que los montos respeten la escala logarítmica calculada en el pipeline[cite: 6, 7].