# mi-repositorio
1	DESCRIPCIÓN GENERAL DEL PROYECTO

Este proyecto se basa en el conjunto de datos del Servicio Nacional de Seguro Médico de Corea y se enfoca en la clasificación de personas en función de sus hábitos de fumar y beber. El objetivo es filtrar y analizar los datos para determinar si una persona fuma o bebe ocasionalmente, lo que puede ser útil en el contexto de la salud y la toma de decisiones clínicas. 

El análisis se realiza en Python y utiliza bibliotecas populares de ciencia de datos como numpy, pandas, matplotlib, seaborn y scikit-learn. El proyecto utiliza Azure Machine Learning para realizar el entrenamiento y despliegue del modelo de clasificación. Además, se realiza el tuneo de hiperparametros del data filtrado, utilizando los modelos que nos proporciona SKlearn, como ser: DecisionTreeClassifier y KNeighborsClassifier para tareas de clasificación de datos. El objetivo principal es encontrar la combinación óptima de hiperparámetros que maximice el rendimiento del modelo.

2	INSTRUCCIONES SOBRE CÓMO INSTALAR Y UTILIZAR EL PROYECTO
2.1	INSTALACIÓN

•	Asegúrate de tener Python instalado en tu sistema.

•	Instala las dependencias necesarias según el proyecto.

2.2	USO

2.2.1	Análisis de Hábitos de Fumar y Beber

•	Ejecuta el archivo para descargar y cargar el conjunto de datos del Servicio Nacional de Seguro Médico de Corea.

•	Realiza el análisis exploratorio de datos para comprender la distribución de las variables y la correlación entre estas.

•	Preprocesa los datos, incluyendo la codificación de variables categóricas y la eliminación de duplicados y valores nulos.

2.2.2	Clasificación de Consumo de Alcohol

•	Instale las dependencias mencionadas anteriormente utilizando: pip install -r requirements.txt.

•	Configure las credenciales de Azure.

•	Descargue los datos de entrenamiento y validación.

•	Ejecute el código proporcionado para configurar el entorno y el trabajo de entrenamiento.

•	Espere a que se complete el trabajo de entrenamiento, puede verificar su estado con el comando ml_client.jobs.stream(returned_job.name).

•	Una vez que el trabajo esté completo, puede obtener el modelo entrenado.

•	Configure un ambiente de despliegue para el modelo utilizando Azure Machine Learning.

2.2.3	Optimización de Bosque Aleatorio

•	Ejecuta el archivo para cargar los datos de tu conjunto de datos. 

•	Sigue las instrucciones para configurar el modelo de Bosque Aleatorio y definir las métricas de evaluación.

•	Utiliza la Búsqueda en Rejilla o la Búsqueda Aleatoria para explorar diferentes combinaciones de hiperparámetros.

•	Examina los resultados de la búsqueda para identificar la mejor configuración de hiperparámetros que maximice la precisión del modelo.

•	Extrae el mejor modelo encontrado y utilízalo para hacer predicciones en nuevos datos.

3	REQUISITOS DEL SISTEMA O DEPENDENCIAS

Para ejecutar, necesitarás:

•	Python 3.x

•	Dependencias específicas mencionadas en las instrucciones de instalación.

4	EJEMPLOS DE USO

•	Análisis de la señal corporal.

•	Clasificación de fumador o bebedor.

5	CÓMO CONTRIBUIR AL PROYECTO

•	Clona el repositorio desde el enlace proporcionado.

•	Realiza tus modificaciones o mejoras en una rama separada.

•	Participa en discusiones y colabora en el desarrollo.

6	CRÉDITOS Y AGRADECIMIENTOS

El proyecto otorga créditos y agradecimientos a los desarrolladores: https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset/code
