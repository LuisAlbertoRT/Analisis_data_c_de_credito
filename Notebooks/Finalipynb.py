# %% [markdown]
# # D) Problema	en	Python:

# %% [markdown]
# #### Descripcion del problema

# %% [markdown]
# La empresa necesita realizar un modelo de fraude. Tu papel como DS es dar una solución de modelo y
# mitigar los riesgos de fraude basada en datos. Los datos para realizar el modelo son datos_fraud.csv.

# %% [markdown]
# 1. Una vez que obtengas los datos, utiliza todos tus conocimientos y todos los pasos que creas
# necesarios para poder entregar un modelo funcional para predecir si un cliente debería de ser
# aprobado o no. Entre los pasos que esperamos ver están:
# - **EDA**
# - **Pre-procesamiento de los datos**
# - **Entrenamiento del modelo**
# - **Testing de modelo**
# - **Una explicación de cómo pondrías este modelo en producción y que tendrías que estarle cuidando con el tiempo**

# %% [markdown]
# 
# 2. Una vez entrenado tu modelo esperamos recibir 3 archivos específicos:
# - **Un Jupyter Notebook que explique todo su proceso de entrenamiento. Aquí mismo es donde vas a incluir los distintos pasos descritos arriba bien documentados para poder entender cómo fuiste generando el modelo.**
# - **CSV de predicciones de tu modelo en testing data. El CSV nada más debe de incluir el ID de solicitante y su score del modelo.**
# - **Un PDF explicando qué punto de corte seleccionarías y por qué.**

# %% [markdown]
# ### 1)Importacion de las librerias

# %%

import json
import os
from datetime import datetime


import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sweetviz as sv

import lightgbm as lgb
import optuna
from xgboost import XGBClassifier


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split,  cross_val_predict
from sklearn.preprocessing import RobustScaler


from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# %% [markdown]
# ### 2) Lectura de datos

# %%
df=pd.read_csv("Archivos_aux/datos_fraud.csv")

# %%
# Contar cuántos transaction_id repetidos son filas 100% idénticas
duplicados_exactos = df.duplicated().sum()
duplicados_id = df.duplicated(subset=["transaction_id"]).sum()

df = df.drop_duplicates().reset_index(drop=True)


# %%
df.head(2)

# %% [markdown]
# ### 3) EDA

# %%
#Se realiza un reporte estadistico con la libreria sweetviz que entrega un html con las distribuciones de cada variable y principales metricas
#------IMPORTANTE---- El codigo esta comentado por que tarda en correr como 4min

#report = sv.analyze(df)
#report.show_html("Salidas/EDA-datos_fraud.html")

# %% [markdown]
# Se hace el analisis estadistico preeliminar de las variables.
# - Las variables no cuentan con un nombre mas que el ID, la ficha de tiempo, el monto de la transaccion, y la clasificacion de si es fraude o no. 
# 
# - Se cuenta con 32 variables sin descripcion con varianzas menores a 2 en la mayor parte de los casos y distribuciones leptocurtucas con datos atipicos en las colas. (Se realizara revision)
# 
# - La variable 31 presenta una varianza mayor a las demas por lo que habra que normalizar para evitar dominancia en el modelo.
# 
# - No se cuenta con datos nulos o faltantes en ninguna columna. 
# 
# - Existen IDs  repetidos siendo el mayor con una repeticion de 18 veces
# 
# - Solo hay 492 casos de fraude de una base de 284,807 siendo menos del 1% de los casos.

# %% [markdown]
# ### 4) Pre-procesamiento de los datos

# %%


#Aumentar el nuvel de precision para pasar a logaritmos
df['amount'] = df['amount'].astype(np.float64)
#Transformación Logarítmica: np.log1p equivale a log(x + 1)
df['amount_log'] = np.log1p(df['amount'])

df['es_uno_o_menos'] = (df['amount'] <= 1.0).astype(int)

#(Validaciones sin costo)
df['es_cero'] = (df['amount'] == 0.0).astype(int)

#Extraer la hora del día (0 a 23)
# Dado que tu timestamp eran segundos transcurridos:
df['hour'] = ((df['timestamp'] / 3600) % 24).astype(int)

#Extraer el día de la transacción (Día 1 vs Día 2)
df['day'] = ((df['timestamp'] / 86400)).astype(int) + 1

#Flag de horario nocturno / riesgo (Las transacciones de madrugada suelen tener mayor índice de fraude)
# Ejemplo: entre las 00:00 y las 06:00 AM
df['is_night_transaction'] = df['hour'].apply(lambda x: 1 if 0 <= x <= 5 else 0)

# Transformación cíclica de la hora (Seno / Coseno)
# Le enseña al modelo que las 23:59 y las 00:01 están consecutivas y son el mismo momento del día
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)


# Separación Estratificada (Train / Test)
X = df.drop(columns=['is_fraud', "transaction_id","amount","timestamp"])  
y = df['is_fraud']

# Separación 80/20 manteniendo el balance de clases
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


# %%

# Aprender los bordes de los deciles SOLO en X_train
_, bin_edges = pd.qcut(
    X_train['amount_log'], 
    q=10, 
    labels=False, 
    retbins=True, 
    duplicates='drop'
)

# Ajustar los extremos a infinito para evitar que datos futuros queden fuera del rango
bin_edges[0] = -float('inf')
bin_edges[-1] = float('inf')

# Aplicar los MISMOS bordes a X_train y a X_test usando pd.cut
X_train['deciles'] = pd.cut(X_train['amount_log'], bins=bin_edges, labels=False, include_lowest=True)
X_test['deciles']  = pd.cut(X_test['amount_log'],  bins=bin_edges, labels=False, include_lowest=True)




# %% [markdown]
# ### 5) Entrenamiento del modelo

# %% [markdown]
# #### 5.1) Modelos iniciales

# %%


# Cálculo del peso de balanceo para desbalance severo
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)


# Definición de Modelos

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=100,
        is_unbalance=True, 
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    ),
}


# %%


# Entrenamiento y Evaluación

results = []

for name, model in models.items():
    print(f"\n=================== Entrenando {name} ===================")
    model.fit(X_train, y_train)

    # Predicción de probabilidades
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Métricas clave para datasets desbalanceados
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(
        y_test, y_pred_proba
    )  # PR-AUC es vital en fraude

    results.append({'Modelo': name, 'ROC-AUC': roc_auc, 'PR-AUC': pr_auc})

    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC (Average Precision): {pr_auc:.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, digits=4))

# Resumen de resultados
df_resumen = pd.DataFrame(results).sort_values(by='PR-AUC', ascending=False)
print("\n=== RESUMEN DE RENDIMIENTO ===")
print(df_resumen.to_string(index=False))

# %% [markdown]
# #### 5.2) Optimizacion de hiperparametros

# %%

# Desactivar logs de Optuna para mantener la consola limpia
optuna.logging.set_verbosity(optuna.logging.WARNING)


# Definición de la métrica objetivo: PR-AUC

pr_auc_scorer = make_scorer(
    average_precision_score, response_method='predict_proba'
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# Objetivos de Optimización para cada Modelo

ratio_real = (y_train == 0).sum() / (y_train == 1).sum()

# OPTIMIZACIÓN XGBOOST
def objective_xgb(trial):
    params = {
        # Arboles y Tasa de Aprendizaje
        'n_estimators': trial.suggest_int('n_estimators', 300, 700),
        'learning_rate': trial.suggest_float(
            'learning_rate', 0.01, 0.08, log=True
        ),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        # Muestreo 
        'subsample': trial.suggest_float('subsample', 0.8, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        # Manejo de Desbalanceo 
        'scale_pos_weight': trial.suggest_float(
            'scale_pos_weight', 50.0, ratio_real, log=True
        ),
        # Controlar overfitting
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss',
    }

    model = XGBClassifier(**params)

    # cross_val_score con tu scorer de PR-AUC
    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring=pr_auc_scorer, n_jobs=None
    )

    return scores.mean()


# OPTIMIZACIÓN LIGHTGBM
def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 800),
        'learning_rate': trial.suggest_float(
            'learning_rate', 0.03, 0.1, log=True
        ),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.8, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'scale_pos_weight': trial.suggest_float(
            'scale_pos_weight', 10.0, 300.0, log=True
        ),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }
    model = lgb.LGBMClassifier(**params)
    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring=pr_auc_scorer, n_jobs=-1
    )
    return scores.mean()


# OPTIMIZACIÓN RANDOM FOREST
def objective_rf(trial):
  params = {
      'n_estimators': trial.suggest_int('n_estimators', 40, 90),
      'max_depth': trial.suggest_int('max_depth', 10, 18),
      'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
      'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
      'max_samples': trial.suggest_float('max_samples', 0.7, 0.95),
      'class_weight': trial.suggest_categorical(
          'class_weight', ['balanced', {0: 1, 1: 10}, {0: 1, 1: 50}]
      ),
      'random_state': 42,
      'n_jobs': -1,
  }

  model = RandomForestClassifier(**params)

  scores = cross_val_score(
      model, X_train, y_train, cv=cv, scoring=pr_auc_scorer, n_jobs=None
  )

  return scores.mean()



# %% [markdown]
# Se comento ya que su tiempo de procesamiento es largo al rededor de dos horas.

# %%


# Ejecución de las Búsquedas (con XGBoost)

#print("Buscando mejores hiperparámetros para XGBoost...")
#study_xgb = optuna.create_study(direction='maximize')
#study_xgb.optimize(objective_xgb, n_trials=25)

#print(f"Mejor PR-AUC en CV (XGBoost): {study_xgb.best_value:.4f}")
#print("Mejores Parámetros:", study_xgb.best_params)

# Re-entrenar el modelo final con los mejores parámetros
#best_xgb = XGBClassifier(**study_xgb.best_params, random_state=42, n_jobs=-1)
#best_xgb.fit(X_train, y_train)

# %%

# Ejecución de las Búsquedas (con LIGHTGBM)

#print("Buscando mejores hiperparámetros para ...")
#study_lgb = optuna.create_study(direction='maximize')
#study_lgb.optimize(objective_lgb, n_trials=25)

#print(f"Mejor PR-AUC en CV (LIGHTGBM): {study_lgb.best_value:.4f}")
#print("Mejores Parámetros:", study_lgb.best_params)

# Re-entrenar el modelo final con los mejores parámetros
#best_lgb = lgb.LGBMClassifier(**study_lgb.best_params, random_state=42, n_jobs=-1)
#best_lgb.fit(X_train, y_train)

# %%

# Ejecución de las Búsquedas (con RF)

#print("Buscando mejores hiperparámetros para RF...")
#study_rf = optuna.create_study(direction='maximize')
#study_rf.optimize(objective_rf, n_trials=25)

#print(f"Mejor PR-AUC en CV (RF): {study_rf.best_value:.4f}")
#print("Mejores Parámetros:", study_rf.best_params)

# Re-entrenar el modelo final con los mejores parámetros
#best_rf = RandomForestClassifier(**study_rf.best_params, random_state=42, n_jobs=-1)
#best_rf.fit(X_train, y_train)

# %% [markdown]
# #### 5.3) Modelos optimizados hardcodeados 

# %% [markdown]
# Los modelos con hiperparametros optimos fueron hardcodeados ya que se corrieron varias simulaciones variando entre variables y el tiempo era demasiado. 

# %%

# Instanciar los 3 modelos optimizados
params_xgb = {
    'n_estimators': 1500,  # Aumentado para que el Early Stopping decida el corte real
    'learning_rate': 0.051649232072272615,
    'max_depth': 5,
    'subsample': 0.9658075634630385,
    'colsample_bytree': 0.6261278549784421,
    'scale_pos_weight': 201.2089646115566,
    'reg_alpha': 0.0013332399017578604,
    'reg_lambda': 0.1381424424249814,
    'gamma': 0.00010212425872946351,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'aucpr',  # Optimiza directamente PR-AUC durante la validación
    'early_stopping_rounds': 20,  # Detiene si no hay mejora en 30 iteraciones consecutivas
}


params_lgb = {
    'n_estimators': 346,
    'learning_rate': 0.030114748590763015,
    'num_leaves': 117,
    'max_depth': 11,
    'min_child_samples': 30,
    'subsample': 0.8625745904349208,
    'colsample_bytree': 0.7568312154591079,
    'scale_pos_weight': 146.75179692375053,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}

# Re-instanciar los modelos con tus configuraciones
params_rf = {
    'n_estimators': 300,
    'max_depth': 14,
    'min_samples_split': 5,
    'min_samples_leaf': 3,
    'max_samples': 0.80,
    'class_weight': {0: 1, 1: 10},
    'ccp_alpha': 0.00005,
    'random_state': 42,
    'n_jobs': -1,
}




# %%


# INSTANCIAR LOS MODELOS COMO VARIABLES best_

best_xgb = XGBClassifier(**params_xgb)
best_lgb = lgb.LGBMClassifier(**params_lgb)
best_rf = RandomForestClassifier(**params_rf)


# %% [markdown]
# #### 5.4) Comparativa y problemas de corte

# %%



# DIVIDIR TRAIN EN TRAIN INTERNO Y VALIDACIÓN

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

modelos = {
    'XGBoost': best_xgb,
    'LightGBM': best_lgb,
    'Random Forest': best_rf,
}


# ENTRENAMIENTO ESPECÍFICO

for nombre, model in modelos.items():
  if nombre == 'XGBoost':
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

  elif nombre == 'LightGBM':
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

  else:
    # Para consistencia con X_tr en el cálculo del Gap de train
    model.fit(X_tr, y_tr)


# EVALUACIÓN Y GRÁFICA

resultados = []
dict_probabilidades = {}

plt.figure(figsize=(9, 6))

#----------------  Evaluar los 3 modelos individuales
for nombre, model in modelos.items():
  # Evaluar train sobre X_tr (los datos que realmente vieron en el .fit)
  y_proba_train = model.predict_proba(X_tr)[:, 1]
  y_proba_test = model.predict_proba(X_test)[:, 1]

  dict_probabilidades[nombre] = (y_proba_train, y_proba_test)

  # Métricas
  pr_train = average_precision_score(y_tr, y_proba_train)
  pr_test = average_precision_score(y_test, y_proba_test)
  roc_test = roc_auc_score(y_test, y_proba_test)

  resultados.append({
      'Modelo': nombre,
      'PR-AUC Train': round(pr_train, 4),
      'PR-AUC Test': round(pr_test, 4),
      'Gap (Train-Test)': round(pr_train - pr_test, 4),
      'ROC-AUC Test': round(roc_test, 4),
  })

  p, r, _ = precision_recall_curve(y_test, y_proba_test)
  plt.plot(r, p, label=f'{nombre} (PR-AUC = {pr_test:.4f})')

#------------------- Evaluar el Ensamble (90% XGB + 10% RF)
W_XGB, W_RF = 0.90, 0.10

y_proba_train_ens = (W_XGB * dict_probabilidades['XGBoost'][0]) + (
    W_RF * dict_probabilidades['Random Forest'][0]
)
y_proba_test_ens = (W_XGB * dict_probabilidades['XGBoost'][1]) + (
    W_RF * dict_probabilidades['Random Forest'][1]
)

pr_train_ens = average_precision_score(y_tr, y_proba_train_ens)
pr_test_ens = average_precision_score(y_test, y_proba_test_ens)
roc_test_ens = roc_auc_score(y_test, y_proba_test_ens)

resultados.append({
    'Modelo': 'Ensamble (XGB+RF)',
    'PR-AUC Train': round(pr_train_ens, 4),
    'PR-AUC Test': round(pr_test_ens, 4),
    'Gap (Train-Test)': round(pr_train_ens - pr_test_ens, 4),
    'ROC-AUC Test': round(roc_test_ens, 4),
})

p_ens, r_ens, _ = precision_recall_curve(y_test, y_proba_test_ens)
plt.plot(
    r_ens,
    p_ens,
    label=f'Ensamble (PR-AUC = {pr_test_ens:.4f})',
    linestyle='--',
    linewidth=2.5,
    color='black',
)


#  IMPRIMIR RESULTADOS Y GRÁFICO FINAL

df_eval = pd.DataFrame(resultados)
print('=' * 65)
print('        EVALUACIÓN FINAL DE FIT Y GENERALIZACIÓN EN TEST        ')
print('=' * 65)
print(df_eval.to_string(index=False))
print('=' * 65)

plt.title('Curvas Precision-Recall Comparativas (Incluye Ensamble)')
plt.xlabel('Recall (Sensibilidad)')
plt.ylabel('Precision (Valor Predictivo Positivo)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# #### 5.5) Busqueda de umbral optimo 

# %%
# Obtener probabilidades predichas del modelo XGBoost 
y_probs_val = best_xgb.predict_proba(X_val)[:, 1]

# Calcular Precision, Recall y Umbrales usando SOLO datos de validacion
precisions, recalls, thresholds = precision_recall_curve(y_val, y_probs_val)

# Calcular F1-Score para encontrar el umbral óptimo
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = float(thresholds[best_idx])

print(f"=== UMBRAL ÓPTIMO (CALCULADO EN VALIDACIÓN): {best_threshold:.4f} ===")
print(f"Precisión esperada: {precisions[best_idx]:.4f}")
print(f"Recall esperado:    {recalls[best_idx]:.4f}")
print(f"F1-Score máximo:     {f1_scores[best_idx]:.4f}\n")

# Obtener probabilidades en el set de TEST y aplicar el umbral prefijado
y_probs_xgb = best_xgb.predict_proba(X_test)[:, 1]
y_pred_optimo = (y_probs_xgb >= best_threshold).astype(int)

# Reporte de Clasificación en TEST con el umbral imparcial
print("=== REPORTE DE CLASIFICACIÓN EN TEST (UMBRAL ÓPTIMO) ===")
print(
    classification_report(
        y_test, y_pred_optimo, target_names=['Legítimo (0)', 'Fraude (1)']
    )
)

# Graficar la Matriz de Confusión en TEST
cm = confusion_matrix(y_test, y_pred_optimo)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Pred. Legítimo', 'Pred. Fraude'],
    yticklabels=['Real Legítimo', 'Real Fraude'],
)
plt.title(
    f'Matriz de Confusión XGBoost en Test\n(Umbral de Decisión = {best_threshold:.4f})'
)
plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción del Modelo')
plt.tight_layout()
plt.show()

# %% [markdown]
# #### 5.6) Mejor modelo

# %% [markdown]
# ##### 5.6.1) Entrenamiento del modelo

# %%


# 1. ESPECIFICACIÓN Y ENTRENAMIENTO DEL MODELO FINAL

print("Entrenando modelo final XGBoost...")

# Se extrae early_stopping_rounds para entrenar sobre todo X_train sin requerir eval_set
best_xgb_params = params_xgb.copy()
best_xgb_params.pop('early_stopping_rounds', None)

final_xgb_model = XGBClassifier(**best_xgb_params)
final_xgb_model.fit(X_train, y_train)


# 2. CÁLCULO DEL UMBRAL ÓPTIMO VÍA OUT-OF-FOLD 

# Generamos predicciones fuera de muestra sobre X_train usando validación cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_probs_train_oof = cross_val_predict(
    final_xgb_model, X_train, y_train, cv=cv, method='predict_proba', n_jobs=-1
)[:, 1]

# Calculamos la curva Precision-Recall usando únicamente etiquetas de TRAIN
precisions, recalls, thresholds = precision_recall_curve(y_train, y_probs_train_oof)

f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores)
OPTIMAL_THRESHOLD = float(thresholds[best_idx])

print(f"\n==================================================")
print(f"   UMBRAL ÓPTIMO CONFIGURADO (OOF TRAIN): {OPTIMAL_THRESHOLD:.4f}")
print(f"   Precisión esperada: {precisions[best_idx]:.4f}")
print(f"   Recall esperado:    {recalls[best_idx]:.4f}")
print(f"   F1-Score máximo:     {f1_scores[best_idx]:.4f}")
print(f"==================================================\n")


# 3. CLASE ENCAPSULADORA PARA DESPLIEGUE EN PRODUCCIÓN (WRAPPER)

class ProductionFraudClassifier:
    """
    Wrapper de producción que encapsula el modelo XGBoost y aplica
    automáticamente el umbral de decisión óptimo en las predicciones.
    """
    def __init__(self, model, threshold: float):
        self.model = model
        self.threshold = threshold
        self.feature_names = getattr(model, 'feature_names_in_', None)

    def predict_proba(self, X):
        """Devuelve las probabilidades continuas de fraude [0.0 - 1.0]."""
        return self.model.predict_proba(X)

    def predict(self, X):
        """Devuelve la clase binaria (0 = Legítimo, 1 = Fraude) usando el umbral óptimo."""
        probs = self.model.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)


# Instanciar el detector listo para producción
fraud_detector_final = ProductionFraudClassifier(
    model=final_xgb_model, 
    threshold=OPTIMAL_THRESHOLD
)


# %% [markdown]
# ##### 5.6.2) Testeo del modelo

# %%


# 4. EVALUACIÓN Y CONFIRMACIÓN DE DESEMPEÑO EN TEST (EVALUACIÓN IMPARCIAL)

y_pred_final = fraud_detector_final.predict(X_test)
y_scores = fraud_detector_final.predict_proba(X_test)[:, 1]


id_test = df.loc[X_test.index, "transaction_id"]


df_predicciones = pd.DataFrame(
    {"id_solicitante": id_test, "score": y_scores}
).reset_index(drop=True)
# Guarda las predicciones
df_predicciones.to_csv("Salidas/predicciones_test.csv", index=False)

print("CSV generado exitosamente con forma:", df_predicciones.shape)
print(df_predicciones.head())


print("=== REPORTE FINAL DE CLASIFICACIÓN EN TEST (UMBRAL APLICADO) ===")
print(classification_report(y_test, y_pred_final, target_names=['Legítimo (0)', 'Fraude (1)']))


# 5. GUARDAR ARTEFACTO COMPLETO A DISCO

os.makedirs("Salidas", exist_ok=True)
file_name = "Salidas/modelo_fraude_xgboost_final.joblib"
joblib.dump(fraud_detector_final, file_name)
print(f"✅ Modelo guardado exitosamente en '{file_name}'")

# %% [markdown]
# #### 5.7) Monitorear problema de sobre ajuste

# %%

# DIVISIÓN INTERNA DE VALIDACIÓN PARA EARLY STOPPING

# Separamos un 15% de X_train para validación interna durante el entrenamiento
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)


# ESPECIFICACIÓN Y ENTRENAMIENTO CON EARLY STOPPING (OPCIÓN 1)

print("Entrenando modelo final XGBoost con Early Stopping (Opción 1)...")

# Conservamos params_xgb tal cual (debe contener 'early_stopping_rounds')
final_xgb_model = XGBClassifier(**params_xgb)

# Entrenamos en X_tr monitoreando el desempeño en X_val
final_xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

print(
    f"✅ Entrenamiento detenido en el árbol {final_xgb_model.best_iteration} (de"
    f" {final_xgb_model.n_estimators} máximos)."
)


# EVALUACIÓN Y VALIDACIÓN DE OVERFITTING (TRAIN VS. TEST)

# Evaluamos en X_tr (Train) y X_test (Test)
y_probs_train = final_xgb_model.predict_proba(X_tr)[:, 1]
y_probs_test = final_xgb_model.predict_proba(X_test)[:, 1]

pr_auc_train = average_precision_score(y_tr, y_probs_train)
pr_auc_test = average_precision_score(y_test, y_probs_test)
roc_auc_train = roc_auc_score(y_tr, y_probs_train)
roc_auc_test = roc_auc_score(y_test, y_probs_test)

gap_pr_auc = pr_auc_train - pr_auc_test
gap_roc_auc = roc_auc_train - roc_auc_test

df_overfitting = pd.DataFrame({
    'Métrica': ['PR-AUC (Principal)', 'ROC-AUC (Secundaria)'],
    'Train': [round(pr_auc_train, 4), round(roc_auc_train, 4)],
    'Test': [round(pr_auc_test, 4), round(roc_auc_test, 4)],
    'Gap (Train - Test)': [round(gap_pr_auc, 4), round(gap_roc_auc, 4)],
})

print('\n' + '=' * 65)
print('        DIAGNÓSTICO DE FIT Y GENERALIZACIÓN (OVERFITTING)       ')
print('=' * 65)
print(df_overfitting.to_string(index=False))
print('=' * 65)

# Veredicto de Overfitting
if gap_pr_auc <= 0.05:
    print('✅ VEREDICTO: Excelente generalización. No hay overfitting crítico.')
elif gap_pr_auc <= 0.10:
    print(
        '⚠️ VEREDICTO: Aceptable. Existe un ligero sobreajuste dentro de los'
        ' límites operativos.'
    )
else:
    print(
        '❌ VEREDICTO: Overfitting severo. Se requiere aumentar la regularización'
        ' (reg_alpha/reg_lambda o min_child_weight).'
    )
print('=' * 65 + '\n')


# CALCULO DEL UMBRAL ÓPTIMO EN VALIDACIÓN 

# Se calcula la probabilidad predicha en X_val y se compara únicamente con y_val
y_probs_val = final_xgb_model.predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, y_probs_val)

f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores)
OPTIMAL_THRESHOLD = float(thresholds[best_idx])

print(f'==================================================')
print(f'   UMBRAL ÓPTIMO CONFIGURADO (EN VALIDACIÓN): {OPTIMAL_THRESHOLD:.4f}')
print(f'   Precisión esperada en Val: {precisions[best_idx]:.4f}')
print(f'   Recall esperado en Val:    {recalls[best_idx]:.4f}')
print(f'   F1-Score máximo en Val:    {f1_scores[best_idx]:.4f}')
print(f'==================================================\n')



# CLASE ENCAPSULADORA PARA DESPLIEGUE EN PRODUCCIÓN (WRAPPER)

class ProductionFraudClassifier:
    """Wrapper de producción que encapsula el modelo XGBoost y aplica
    automáticamente el umbral de decisión óptimo en las predicciones.
    """

    def __init__(self, model, threshold: float):
        self.model = model
        self.threshold = threshold
        self.feature_names = getattr(model, 'feature_names_in_', None)

    def predict_proba(self, X):
        """Devuelve las probabilidades continuas de fraude [0.0 - 1.0]."""
        return self.model.predict_proba(X)

    def predict(self, X):
        """Devuelve la clase binaria (0 = Legítimo, 1 = Fraude) usando el umbral óptimo."""
        probs = self.model.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)


# Instanciar el detector listo para producción
fraud_detector_final = ProductionFraudClassifier(
    model=final_xgb_model, threshold=OPTIMAL_THRESHOLD
)


# REPORTE FINAL DE CLASIFICACIÓN EN TEST (EVALUACIÓN IMPARCIAL)

y_pred_final = fraud_detector_final.predict(X_test)

print('=== REPORTE FINAL DE CLASIFICACIÓN EN TEST (UMBRAL APLICADO) ===')
print(
    classification_report(
        y_test, y_pred_final, target_names=['Legítimo (0)', 'Fraude (1)']
    )
)



os.makedirs("Salidas", exist_ok=True)
file_name = 'Salidas/modelo_fraude_xgboost_final.joblib'
joblib.dump(fraud_detector_final, file_name)
print(f"✅ Modelo guardado exitosamente en '{file_name}'")

# %% [markdown]
# ### 6) Reporte de mejor modelo

# %%

# Set global de estilo para los gráficos del reporte
plt.style.use(
    'seaborn-v0_8-whitegrid'
    if 'seaborn-v0_8-whitegrid' in plt.style.available
    else 'default'
)
sns.set_context('notebook', font_scale=1.1)


# EXTRACCIÓN Y CÁLCULO DE MÉTRICAS CONSOLIDADAS

# Inferencia sobre los conjuntos de datos de Train e In-Sample Test
y_probs_tr = fraud_detector_final.predict_proba(X_tr)[:, 1]
y_probs_te = fraud_detector_final.predict_proba(X_test)[:, 1]

# Predicción aplicando el umbral óptimo definido
y_pred_te = (y_probs_te >= OPTIMAL_THRESHOLD).astype(int)

# Métricas discriminativas universales
pr_auc_tr = average_precision_score(y_tr, y_probs_tr)
pr_auc_te = average_precision_score(y_test, y_probs_te)
roc_auc_tr = roc_auc_score(y_tr, y_probs_tr)
roc_auc_te = roc_auc_score(y_test, y_probs_te)

# Métricas puntuales en Test con el umbral óptimo aplicado
report_dict = classification_report(y_test, y_pred_te, output_dict=True)
prec_1 = report_dict['1']['precision']
rec_1 = report_dict['1']['recall']
f1_1 = report_dict['1']['f1-score']
f2_1 = fbeta_score(y_test, y_pred_te, beta=2)

# Desglose de Matriz de Confusión
cm = confusion_matrix(y_test, y_pred_te)
tn, fp, fn, tp = cm.ravel()
total_legit = tn + fp
total_fraud = fn + tp


# TABLAS DE ESTADÍSTICA EXECUTIVA PARA EL REPORTE

print("=" * 75)
print("            INFORME ESTADÍSTICO DE DESEMPEÑO DEL MODELO FINAL")
print("=" * 75)

# A.-------------------- Tabla de Fit y Generalización
df_fit = pd.DataFrame({
    'Métrica Evaluada': ['PR-AUC (Principal)', 'ROC-AUC (Secundaria)'],
    'Entrenamiento (X_tr)': [f'{pr_auc_tr:.4f}', f'{roc_auc_tr:.4f}'],
    'Prueba (X_test)': [f'{pr_auc_te:.4f}', f'{roc_auc_te:.4f}'],
    'Gap Generalización': [
        f'{(pr_auc_tr - pr_auc_te):.4f}',
        f'{(roc_auc_tr - roc_auc_te):.4f}',
    ],
})
print('\n[TABLA 1: DIAGNÓSTICO DE OVERFITTING Y GENERALIZACIÓN]')
print(df_fit.to_string(index=False))

# B.-------------------- Tabla de Métricas Operativas en Test
df_metrica_operativa = pd.DataFrame({
    'Métrica de Negocio': [
        'Umbral de Decisión Aplicado',
        'Precisión en Fraude (Precision)',
        'Cobertura de Fraude (Recall)',
        'F1-Score (Balance Precisión/Recall)',
        'F2-Score (Mayor peso al Recall de Fraude)',
        'Tasa de Falsas Alarmas (FPR)',
    ],
    'Valor Obtenido': [
        f'{OPTIMAL_THRESHOLD:.4f}',
        f'{prec_1:.4f} ({prec_1:.1%})',
        f'{rec_1:.4f} ({rec_1:.1%})',
        f'{f1_1:.4f}',
        f'{f2_1:.4f}',
        f'{(fp / total_legit):.4%} ({fp} casos)',
    ],
})
print('\n[TABLA 2: RESUMEN DE DESEMPEÑO EN PRODUCCIÓN (TEST)]')
print(df_metrica_operativa.to_string(index=False))

# C.------------------------- Tabla Desagregada de Impacto Financiero/Operativo
df_impacto = pd.DataFrame({
    'Clase Real': ['Legítimo (0)', 'Legítimo (0)', 'Fraude (1)', 'Fraude (1)'],
    'Predicción Modelo': [
        'Aprobado (0)',
        'Bloqueado (1)',
        'No Detectado (0)',
        'Capturado (1)',
    ],
    'Tipo de Resultado': [
        'Verdadero Negativo (TN)',
        'Falso Positivo (FP)',
        'Falso Negativo (FN)',
        'Verdadero Positivo (TP)',
    ],
    'Conteo Casos': [tn, fp, fn, tp],
    'Porcentaje por Clase': [
        f'{(tn/total_legit):.2%}',
        f'{(fp/total_legit):.2%}',
        f'{(fn/total_fraud):.2%}',
        f'{(tp/total_fraud):.2%}',
    ],
})
print('\n[TABLA 3: DESGLOSE OPERATIVO DE TRANSACCIONES EN TEST]')
print(df_impacto.to_string(index=False))
print('=' * 75 + '\n')


# GENERACIÓN DE VISUALIZACIONES PARA DOCUMENTACIÓN (DASHBOARD)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# --- Gráfico 1: Matriz de Confusión ---
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False,
    ax=axes[0, 0],
    annot_kws={'size': 14, 'weight': 'bold'},
    xticklabels=['Legítimo (0)', 'Fraude (1)'],
    yticklabels=['Legítimo (0)', 'Fraude (1)'],
)
axes[0, 0].set_title(
    '1. Matriz de Confusión en Test', fontsize=13, fontweight='bold', pad=10
)
axes[0, 0].set_xlabel('Predicción del Modelo')
axes[0, 0].set_ylabel('Clase Real')

# --- Gráfico 2: Curva Precision-Recall ---
prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_probs_te)
axes[0, 1].plot(
    rec_curve,
    prec_curve,
    color='#1f77b4',
    lw=2.5,
    label=f'PR-AUC Test = {pr_auc_te:.4f}',
)
axes[0, 1].plot(
    rec_1,
    prec_1,
    marker='o',
    markersize=9,
    color='red',
    label=f'Umbral Óptimo ({OPTIMAL_THRESHOLD:.4f})',
)
axes[0, 1].set_title(
    '2. Curva Precision-Recall', fontsize=13, fontweight='bold', pad=10
)
axes[0, 1].set_xlabel('Recall (Cobertura de Fraude)')
axes[0, 1].set_ylabel('Precision (Exactitud de Alerta)')
axes[0, 1].legend(loc='lower left')

# --- Gráfico 3: Importancia de Variables (Feature Importance - Top 10) ---
raw_xgb = fraud_detector_final.model
importances = pd.Series(
    raw_xgb.feature_importances_, index=X_tr.columns
).sort_values(ascending=False)[:10]

sns.barplot(
    x=importances.values,
    y=importances.index,
    palette='Blues_r',
    ax=axes[1, 0],
    hue=importances.index,
    legend=False,
)
axes[1, 0].set_title(
    '3. Top 10 Variables clave (Gain Weight)',
    fontsize=13,
    fontweight='bold',
    pad=10,
)
axes[1, 0].set_xlabel('Importancia Relativa')

# --- Gráfico 4: Distribución de Probabilidades por Clase (Con Leyenda Completa) ---
df_probs = pd.DataFrame({
    'Probabilidad': y_probs_te,
    'Clase': pd.Series(y_test).map({0: 'Legítimo (0)', 1: 'Fraude (1)'}),
})

sns.histplot(
    data=df_probs,
    x='Probabilidad',
    hue='Clase',
    bins=50,
    element='step',
    stat='density',
    common_norm=False,
    palette={'Legítimo (0)': '#2ca02c', 'Fraude (1)': '#d62728'},
    alpha=0.4,
    ax=axes[1, 1],
)

axes[1, 1].axvline(
    OPTIMAL_THRESHOLD,
    color='black',
    linestyle='--',
    linewidth=2,
    label=f'Umbral ({OPTIMAL_THRESHOLD:.4f})',
)
axes[1, 1].set_yscale('log')
axes[1, 1].set_xlim(-0.02, 1.02)
axes[1, 1].set_title(
    '4. Separación de Probabilidades (Escala Log)',
    fontsize=13,
    fontweight='bold',
    pad=10,
)
axes[1, 1].set_xlabel('Probabilidad Predicha de Fraude')
axes[1, 1].set_ylabel('Densidad Logarítmica')

# Forzar a Matplotlib a consolidar la leyenda de Seaborn + la axvline
handles, labels = axes[1, 1].get_legend_handles_labels()
axes[1, 1].legend(handles=handles, labels=labels, loc='upper center')

# %% [markdown]
# #### 7) Guardado de metadatos para construccion de la model card

# %%


# 1. CONSTRUCCIÓN DEL DICCIONARIO DE METADATOS

metadata = {
    "model_info": {
        "model_name": "Fraud_Detector_XGBoost",
        "model_version": "1.0.0",
        "description": "Modelo supervisado para detección de fraude transaccional altamente desbalanceado.",
        "author": "Equipo de Data Science / Analytics",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "framework": "xgboost / scikit-learn",
        "algorithm": fraud_detector_final.model.__class__.__name__,
    },
    "decision_threshold": {
        "optimal_threshold": float(OPTIMAL_THRESHOLD),
        "threshold_metric_criterion": "F2-Score / Precision-Recall Optimization",
    },
    "data_schema": {
        "target_variable": "es_fraude",
        "num_features": len(X_tr.columns),
        "feature_names": list(X_tr.columns),
    },
    "hyperparameters": {
        k: (float(v) if isinstance(v, (np.float32, np.float64)) else v)
        for k, v in fraud_detector_final.model.get_params().items()
    },
    "performance_metrics": {
        "test_set": {
            "pr_auc": float(pr_auc_te),
            "roc_auc": float(roc_auc_te),
            "precision": float(prec_1),
            "recall": float(rec_1),
            "f1_score": float(f1_1),
            "f2_score": float(f2_1),
            "false_positive_rate": float(fp / total_legit),
        },
        "confusion_matrix_test": {
            "true_negatives_TN": int(tn),
            "false_positives_FP": int(fp),
            "false_negatives_FN": int(fn),
            "true_positives_TP": int(tp),
        },
        "generalization_gap": {
            "pr_auc_gap": float(pr_auc_tr - pr_auc_te),
            "roc_auc_gap": float(roc_auc_tr - roc_auc_te),
        },
    },
    "feature_importance_top10": importances.to_dict(),
}


# 2. ESCRITURA EN ARCHIVO JSON

os.makedirs("Salidas", exist_ok=True)
json_filepath = "Salidas/model_metadata.json"

with open(json_filepath, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)

print(f"✅ Archivo de metadatos guardado exitosamente en: '{json_filepath}'")


