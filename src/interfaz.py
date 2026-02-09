import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

st.set_page_config(page_title = "Predictor de Fuga de CLientes",
                   layout = "wide")

# Cargamos el modelo y las columnas
@st.cache_resource
def cargar_datos():
    ruta_script = os.path.dirname(__file__)
    ruta_raiz = os.path.abspath(os.path.join(ruta_script, ".."))
    path_modelo = os.path.join(ruta_raiz, 'models', 'modelo_churn.pkl')
    path_cols = os.path.join(ruta_raiz, 'models', 'columnas_modelo.pkl')
    try:
        modelo = joblib.load(path_modelo)
        columnas = joblib.load(path_cols)
        return modelo, columnas
    except FileNotFoundError:
        st.error(f"No se encontró la carpeta 'models' en la raíz: {ruta_raiz}")
        st.stop()

# Ejecución
modelo, columnas_modelo = cargar_datos()


# INTERFAZ GRÁFICA
st.title("PREDICTOR PÉRDIDA DE CLIENTES")
st.markdown("""
            Esta herramienta empple XGBoost para predecir la probabilidad
            de que un cliente abandone la compañía.
            """)

# PANEL LATERAL
st.sidebar.header("Perfil del Cliente")

def caract_cliente():
    # Variables numéricas
    tenencia = st.sidebar.slider('Meses de permanencia', 1, 72, 12)
    factura_mensual = st.sidebar.number_input('Factura Mensual ($)', min_value = 18.0, max_value = 120.0, value = 70.0)
    factura_total = st.sidebar.number_input('Total Facturado ($)', min_value=18.0, max_value=9000.0, value = factura_mensual * tenencia)

    # Variable Categóricas (de mayor importancia)
    contrato = st.sidebar.selectbox('Tipo de Contrato', ['Month-to-month' , 'One year' , 'Two years'])
    servicio_internet = st.sidebar.selectbox('Servicio de Internet', ['DSL' , 'Fiber optic' , 'No'])
    metodo_pago = st.sidebar.selectbox('Método de Pago', ['Electronic check' , 'Mailed check' , 'Bank transfer (automatic)' , 'Credit card (automatic)'])
    seguridad_online = st.sidebar.selectbox('Seguridad Online' , ['Yes' , 'No' , 'No internet service'])
    soporte_tecnico = st.sidebar.selectbox('Soporte Técnico', ['Yes', 'No', 'No internet service'])
    factura_dig = st.sidebar.selectbox('Factura Digital (Paperless)', ['Yes', 'No'])

    # Se crea un diccionario con los datos sin procesar
    datos = {
        'tenure' : tenencia,
        'MonthlyCharges' : factura_mensual,
        'TotalCharges' : factura_total,
        'Contract' : contrato,
        'InternetService' : servicio_internet,
        'PaymentMethod' : metodo_pago,
        'OnlineSecurity' : seguridad_online,
        'TechSupport' : soporte_tecnico,
        'PaperlessBilling': factura_dig,
        # Añadimos valores por defecto para el resto de variables menos críticas
        'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No', 
        'PhoneService': 'Yes', 'MultipleLines': 'No', 'OnlineBackup': 'No',
        'DeviceProtection': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No',
        'gender': 'Male'
    }
    return pd.DataFrame(datos,index = [0])

caract = caract_cliente()

# Mostrar los datos elegidos por el uauario
st.subheader('Datos del Cliente: ')
st.dataframe(caract)

# PREPROCESAMIENTO
# Pasamos el input del usuario al formato del modelo

# Primero aplicamos One-Hot Encoding a los datos de entrada
caract_oneHot = pd.get_dummies(caract)
# Alineamos con las columnas del modelo origibal
caract_oneHot_final = caract_oneHot.reindex(columns = columnas_modelo, fill_value=0)

# PREDICCIÓN
if st.button('Calacular Riesgo de Fuga'):
    #Predeccimos las clases (0 y 1) y su probabilidad
    prediccion = modelo.predict(caract_oneHot_final)
    prob = modelo.predict_proba(caract_oneHot_final)

    riesgo_abandono = prob[0][1] # Probabilidad de que sea 1

    st.subheader('Resultado del análiss: ')
    # Ver resultado
    col1, col2 = st.columns([1,2])
    with col1:
        if riesgo_abandono > 0.5:
            st.error(f"PELIGRO!!! ALERTA DE FUGA \n\nRiesgo: {riesgo_abandono: .1%}")
        else:
            st.success(f"Cliente Seguro \n\nRiesgo: {riesgo_abandono: .1%}")

    with col2:
        st.write("Probabilidad de Abandono: ")
        st.progress(int(riesgo_abandono * 100))

    st.write("**Insights Automáticos:**")
    if caract['Contract'].iloc[0] == 'Month-to-month':
        st.write("El contrato **Mes a Mes** aumenta drásticamente el riesgo.")
    if caract['InternetService'].iloc[0] == 'Fiber optic':
        st.write("Los usuarios de **Fibra Óptica** suelen ser más exigentes y propensos a cambiar.")
    if caract['tenure'].iloc[0] < 12:
        st.write("Los clientes **nuevos (<1 año)** son los más volátiles.")