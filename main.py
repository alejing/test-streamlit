import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st

def main_app():
    file_path = 'https://raw.githubusercontent.com/alejing/set-datos-visualizacion/refs/heads/main/insurance.csv'
    data = pd.read_csv(file_path)

    st.title("Predicción de Costos Médicos")
    st.write("Esta aplicación utiliza un modelo de aprendizaje automático para predecir los costos médicos estimados "
             "de un paciente en función de variables clave como edad, índice de masa corporal (IMC), sexo, "
             "número de hijos, estado de tabaquismo y región geográfica. Introduce los datos solicitados y obtén una "
             "estimación personalizada, diseñada para ofrecerte una idea general del posible costo médico en función "
             "de factores individuales.")
    st.warning("**Nota:** Esta predicción es solo una estimación y no representa una evaluación médica real. Maneje "
               "la información con su médico de confianza.")
    st.subheader("Por favor, ingresa los datos que le serán solicitados a continuación.")
    # Solicitar los datos de entrada al usuario
    age = st.number_input("Edad", min_value=18, max_value=120, value=30)
    bmi = st.number_input("Índice de Masa Corporal (BMI)", min_value=0.0, max_value=100.0, value=25.0)
    sex = st.selectbox("Sexo", ["Masculino", "Femenino"])
    children = st.number_input("Número de Hijos", min_value=0, max_value=5, value=0)
    smoker = st.selectbox("¿Fumador?", ["Sí", "No"])
    region = st.selectbox("Región", ["Northeast", "Northwest", "Southeast", "Southwest"])

    # Convertir los valores de texto en valores numéricos (si es necesario para el modelo)
    sex = 1 if sex == "Femenino" else 0
    smoker = 1 if smoker == "Sí" else 0
    if region == "Northeast":
        region = 4
    elif region == "Northwest":
        region = 1
    elif region == "Southeast":
        region = 2
    else:
        region = 3

    # Cargar el modelo entrenado
    with open('modelo_rf.pkl', 'rb') as file:
        model = pickle.load(file)

    nombres_columnas = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    mae_rf = 2533.1799395613193

    # Botón para realizar la predicción
    if st.button("Predecir Costos Médicos"):
        # Realizando la predicción
        nuevo_registro = pd.DataFrame([[age, sex, bmi, children, smoker, region]], columns=nombres_columnas)

        # Ignorar advertencias específicas
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            pred = model.predict(np.array(nuevo_registro).reshape(1, -1))

        # Cálculo de rango estimado
        rango_inferior = pred[0] - mae_rf
        rango_superior = pred[0] + mae_rf

        # Formatear el mensaje
        mensaje = (
            f"### 📈 Predicción de Costos Médicos\n"
            f"El modelo estima unos **cargos médicos** de **\\${pred[0]:,.2f}** dólares, "
            f"con un **margen de error promedio** de **±\\${mae_rf:,.2f}** dólares.\n\n"
            f"💡 Esto sugiere que el valor real probablemente esté en un rango de entre "
            f"**\\${rango_inferior:,.2f}** y **\\${rango_superior:,.2f}** dólares."
        )

        # Mostrar el mensaje
        st.info(mensaje)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main_app()
