#!/usr/bin/env python
# coding: utf-8

# In[21]:


# 1. Cargar librerías y datos
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, RocCurveDisplay
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[22]:


# Importar la base
df = pd.read_excel("df_FS_hogar_binario_com_agrupados4.3%.xlsx")
df.head()


# In[23]:


# Nombramos las variables
X = df.drop(columns="Clase_num")
y = df["Clase_num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)


# ## Balanceo y escalado 

# In[24]:


#Balanceo
smote = SMOTE(random_state=42, k_neighbors=3)
rus = RandomUnderSampler(random_state=42)

X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
X_train_bal, y_train_bal = rus.fit_resample(X_train_res, y_train_res)

# 2.2 Escalado de variables numéricas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)


# ##  Modelos y parámetros

# In[25]:


models = {
    'XGBoost': XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42),

}

param_grids = {
    'XGBoost': {'clf__n_estimators': [50], 'clf__max_depth': [3, 4], 'clf__learning_rate': [0.05, 0.1], 'clf__reg_alpha':[0.5, 1]},

}


# In[26]:


# BLOQUE 4: Entrenamiento, evaluación, matriz y reporte

def train_and_eval(model_name, estimator, param_grid):
    pipe = ImbPipeline([
        # No usas smote aquí porque ya está hecho antes
        ('clf', estimator)
    ])
    grid = GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), n_jobs=-1)
    grid.fit(X_train_scaled, y_train_bal)
    best_model = grid.best_estimator_
    y_pred_test = best_model.predict(X_test_scaled)
    y_pred_train = best_model.predict(X_train_scaled)
    y_proba_test = best_model.predict_proba(X_test_scaled)[:, 1]
    y_proba_train = best_model.predict_proba(X_train_scaled)[:, 1]
    # --- Métricas
    print(f"\n=== {model_name} ===")
    print(f"Mejores hiperparámetros: {grid.best_params_}")
    print(f"Accuracy (train): {accuracy_score(y_train_bal, y_pred_train):.3f}")
    print(f"Accuracy (test): {accuracy_score(y_test, y_pred_test):.3f}")
    print(f"Gap absoluto: {abs(accuracy_score(y_train_bal, y_pred_train) - accuracy_score(y_test, y_pred_test)):.3f}")
    if abs(accuracy_score(y_train_bal, y_pred_train) - accuracy_score(y_test, y_pred_test)) > 0.10:
        print("Diagnóstico: OVERFITTING")
    else:
        print("Diagnóstico: OK")
    print(f"ROC AUC (test): {roc_auc_score(y_test, y_proba_test):.3f}")
    print(f"Recall (test): {recall_score(y_test, y_pred_test):.3f}")
    print(f"F1 (test): {f1_score(y_test, y_pred_test):.3f}")
    print("\nReporte de clasificación (test):")
    print(classification_report(y_test, y_pred_test))
    # --- Matriz de confusión estilo imagen
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, cmap='Greens')
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()
    # --- Curva ROC
    RocCurveDisplay.from_predictions(y_test, y_proba_test)
    plt.title(f"{model_name} ROC Curve")
    plt.show()
    plt.show()
    return best_model



# In[27]:


best_model = train_and_eval("XGBoost", models["XGBoost"], param_grids["XGBoost"])


# In[28]:


for name, model in models.items():
    train_and_eval(name, model, param_grids[name])


# In[29]:


import joblib


# In[30]:


# Guardar el mejor modelo  (ya entrenado)
joblib.dump(best_model, "modelo_xgb_interfaz.pkl")


# In[31]:


# Guarda el escalador usado en el bloque 2.2
joblib.dump(scaler, "scaler_xgb.pkl")


# In[32]:


# Guarda los nombres de las columnas para mantener el orden exacto
joblib.dump(list(X.columns), "feature_names.pkl")

print("✅ Modelo, scaler y columnas guardadas correctamente.")
print("Archivos exportados:")
print("- modelo_xgb_interfaz.pkl")
print("- scaler_xgb.pkl")
print("- feature_names.pkl")

