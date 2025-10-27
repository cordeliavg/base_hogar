#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------
# CARGAR MODELO Y OBJETOS
# ---------------------------
modelo = joblib.load("modelo_xgb_interfaz.pkl")
scaler = joblib.load("scaler_xgb.pkl")
feature_names = joblib.load("feature_names.pkl")

# ---------------------------
# CONFIGURACIÓN DE LA APP
# ---------------------------
st.set_page_config(page_title="Predicción de Éxito de Tienda OXXO", layout="centered")
st.title("🟠 Predicción de Éxito de Tienda OXXO")
st.write("Completa los datos del entorno para predecir si una tienda será **exitosa o no**.")

st.markdown("---")

# ---------------------------
# FORMULARIO DE VARIABLES
# ---------------------------

st.subheader("Datos del entorno del hogar 🏠")

MTS2VENTAS_NUM = st.number_input("Metros Cuadrados de Ventas", min_value=0.0, step=0.1)
PUERTASREFRIG_NUM = st.number_input("Cant. Puertas de Refrigeración", min_value=0)
CAJONESESTACIONAMIENTO_NUM = st.number_input("Cant. Cajones de Estacionamiento", min_value=0)
LON = st.number_input("Longitud (coordenada geográfica)", step=0.000001, format="%.6f")
LAT = st.number_input("Latitud (coordenada geográfica)", step=0.000001, format="%.6f")
Salud = st.number_input("Clínicas y Hospitales cercanos", min_value=0)
Comercios = st.number_input("Comercios cercanos", min_value=0)
viv_100m = st.number_input("Viviendas a 100 metros", min_value=0)
CompetenciaDirecta_personal = st.number_input("Competencia Directa", min_value=0)
Bancos_Cajeros_personal = st.number_input("Bancos y Cajeros", min_value=0)

st.markdown("---")

# ---------------------------
# BOTÓN DE PREDICCIÓN
# ---------------------------
if st.button("🔍 Predecir Éxito"):
    try:
        # Crear DataFrame con el orden de features correcto
        input_data = pd.DataFrame([[
            MTS2VENTAS_NUM,
            PUERTASREFRIG_NUM,
            CAJONESESTACIONAMIENTO_NUM,
            LON,
            Salud,
            Comercios,
            viv_100m,
            LAT,
            CompetenciaDirecta_personal,
            Bancos_Cajeros_personal
        ]], columns=feature_names)

        # Escalar datos
        input_scaled = scaler.transform(input_data)

        # Predicción
        pred = modelo.predict(input_scaled)[0]
        prob = modelo.predict_proba(input_scaled)[0][1]

        # Mostrar resultado
        st.subheader("🧠 Resultado del modelo")
        if pred == 1:
            st.success(f"La tienda tiene **alta probabilidad de ser EXITOSA** 🎯 (Confianza: {prob*100:.2f}%)")
        else:
            st.error(f"La tienda tiene **baja probabilidad de ser exitosa** ⚠️ (Confianza: {(1-prob)*100:.2f}%)")

        # Mostrar accuracy del modelo
        if hasattr(modelo, "best_score_"):
            acc = modelo.best_score_
        elif hasattr(modelo, "score"):
            acc = modelo.score(input_scaled, [pred])  # se calcula rápido
        else:
            acc = None

        if acc is not None:
            st.info(f"**Accuracy del modelo:** {acc:.2f}")
        else:
            st.info("**Accuracy no disponible en el modelo cargado.**")

    except Exception as e:
        st.error(f"Ocurrió un error al procesar la predicción: {e}")

