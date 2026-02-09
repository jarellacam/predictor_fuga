# PREDICTOR DE FUGA DE CLIENTES (CHURN)

## DESCRIPCIÓN DEL PROYECTO

Este es un proyecto práctico de **Machine Learning** donde se aplican técnicas de clasificación para resolver un problema de negocio real: **predecir si un cliente va a abandonar una compañía telefónica.**

El objetivo no ha sido únicamente "entrenar un modelo", sino construir el flujo completo: desde que llegan los datos "sucios" hasta que tienes una aplicación web funcional que te da probabilidades en tiempo real.

## TECNOLOGÍAS EMPLEADAS

Se ha utilizado el stack estándar de Ciencia de Datos en Python:

* **Pandas & NumPy:** Para manipulación y limpieza de datos.
* **Scikit-Learn:** Para el preprocesamiento (One-Hot Encoding) y métricas.
* **XGBoost:** El algoritmo estrella. Lo elegí porque funciona increíblemente bien con datos tabulares.
* **Streamlit:** Para crear la interfaz web (Frontend) de forma rápida y eficiente en Python.
* **Joblib:** Para guardar el modelo entrenado y poder usarlo después en la app.

## ¿QUÉ HACE EL MODELO?

El proyecto sigue estos pasos:

1.  **Limpieza de Datos (EDA):** Se analizó el dataset *Telco Customer Churn*. Se pudo observar que variables como el tipo de contrato o tener fibra óptica influyen mucho.
2.  **Ingeniería de Características:** Se transformaron las variables de texto (como "Male/Female" o "Yes/No") en números que el ordenador pueda entender.
3.  **Entrenamiento:** Se entrenó un clasificador **XGBoost**.
    * *Nota:* Como había muchos más clientes que se quedaban que los que se iban, se tubo que balancear el modelo (usando `scale_pos_weight`) para que aprendiera a detectar bien las fugas.
4.  **Despliegue:** Se creó una interfaz donde el usuario mover sliders y ver cómo cambia el riesgo de fuga al instante.

## RESULTADOS OBTENNIDOS

Para este problema, lo más importante es **no dejar escapar a nadie**. Por eso, se optimizó el **Recall** (Sensibilidad).

| Métrica | Resultado | ¿Qué significa? |
| :--- | :--- | :--- |
| **Recall (Churn)** | **82%** | De cada 10 clientes que se van a ir, el modelo detecta a más de 8. |
| **Accuracy** | 75% | El porcentaje total de aciertos del sistema. |

### CONCLUSIONES DEL ANÁLISIS
Gracias a la importancia de variables del modelo, aprendí que:
* Los contratos **mensuales** son el mayor peligro (la gente se va más).
* La **antigüedad** protege: cuantos más años llevan, menos probable es que se vayan.
* El servicio de **Fibra Óptica** tiene una tasa de abandono curiosamente alta.

## ESTRUCTURA DEL PROYECTO

Organización limpia de carpetas siguiendo buenas prácticas:

```text
PREDICTOR/
├── data/                  # El CSV original
├── models/                # Aquí se guardan los archivos .pkl (el "cerebro" del modelo)
├── src/                   # Código de la aplicación web
│   └── interfaz.py        
├── preprocesamiento.ipynb # Mi notebook con todo el código de entrenamiento
├── requirements.txt       # Librerías necesarias para ejecutarlo
└── README.md              
```

# CÓMO PROBAR EL PROYECTO

1. Clona el repo:
    ````
    git clone [https://github.com/TU_USUARIO/predictor_fuga.git](https://github.com/TU_USUARIO/predictor_fuga.git)
    cd churn-predictor
    ````
2. Instala las librerías:
    ````
    pip install -r requirements.txt
    ````
3. Lanza la App:
    ````
    streamlit run src/interfaz.py
    ````
¡Y listo! Se te abrirá una pestaña en el navegador para jugar con el modelo.

Autor: Juan Arellano Cameo
Linkedin: www.linkedin.com/in/juann-arellano