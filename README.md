[README_Spark_MLlib.md](https://github.com/user-attachments/files/32218448/README_Spark_MLlib.md)
# Modelo_predictivo_en_Apache_Spark_MLlib
# Detección de Transacciones Riesgosas con Apache Spark MLlib

Modelo predictivo de clasificación binaria construido íntegramente en **Apache Spark MLlib** para identificar transacciones de venta potencialmente riesgosas en una red de sucursales, entregando además una probabilidad de riesgo por operación para apoyar decisiones operacionales.

## 🎯 Problema

A partir de un registro de ventas simuladas (sucursal, producto, cantidad, precio, monto y fecha/hora), el objetivo es detectar automáticamente qué transacciones presentan un patrón riesgoso, de modo que el área operacional pueda priorizar su revisión en lugar de auditar todas las operaciones manualmente.

**Regla operacional para la etiqueta (`label`):** una transacción se marca como riesgosa (`1`) si el monto total supera los $7.000 **o** si ocurre en horario de madrugada (00:00–05:59).

> ⚠️ La etiqueta es **sintética**, construida según la regla operacional del enunciado. Debe entenderse como una aproximación inicial del riesgo, no como un histórico real de fraude.

## 🛠️ Stack y metodología

- **Motor:** Apache Spark 4.0.4 (PySpark) — `SparkSession`, procesamiento distribuido y `cache()` sobre el dataset reutilizado
- **Ingesta:** lectura de CSV con separador `;` y encoding `ISO-8859-1`, con inferencia de esquema
- **Limpieza:** eliminación de filas vacías, casteo de variables numéricas, parseo robusto de timestamps con `try_to_timestamp`
- **Feature engineering:** extracción de la variable `Hora` desde el timestamp; construcción de la etiqueta binaria con lógica condicional `when/otherwise`
- **Preprocesamiento:** `StringIndexer` (Sucursal, Producto) + `VectorAssembler`, encapsulados en un `Pipeline` de Spark ML
- **Modelo:** `RandomForestClassifier` (100 árboles, profundidad máxima 5, semilla fija 42)
- **Evaluación:** `MulticlassClassificationEvaluator` (Accuracy, F1) y `BinaryClassificationEvaluator` (área bajo la curva ROC), más matriz de confusión e importancia de variables

### ¿Por qué Random Forest?

Entre los algoritmos disponibles (Logistic Regression, Decision Tree, Random Forest) se eligió Random Forest porque resuelve clasificación binaria, captura relaciones no lineales entre las variables de la transacción, es más robusto que un árbol individual al combinar múltiples estimadores, tolera variables en escalas distintas y permite tanto obtener **probabilidades de riesgo** como analizar la **importancia de cada variable**.

## 📊 Resultados

**Datos:** 999 registros iniciales → 180 registros válidos tras limpieza (110 normales / 70 riesgosos). División 70/30 → 128 entrenamiento, 52 prueba.

| Métrica | Valor |
|---|---|
| Accuracy | 0.9808 |
| F1-Score | 0.9807 |
| Área bajo ROC | 0.9984 |

**Matriz de confusión (set de prueba):**

| | Predicho 0 | Predicho 1 |
|---|---|---|
| **Real 0** | 33 | 0 |
| **Real 1** | 1 | 18 |

Un solo falso negativo, sin falsos positivos.

**Importancia de las variables:**

| Variable | Importancia |
|---|---|
| Monto_Total | 0.4578 |
| Hora | 0.3834 |
| Precio_Unitario | 0.0932 |
| Cantidad | 0.0306 |
| Producto_Index | 0.0242 |
| Sucursal_Index | 0.0108 |

El modelo recupera correctamente las dos dimensiones que definen la regla de negocio (monto y horario), lo que confirma coherencia entre la etiqueta y lo aprendido.

Además de la predicción binaria, el pipeline expone la **probabilidad de riesgo** por transacción (extraída con `vector_to_array`), lo que permite ordenar las operaciones por prioridad de revisión en lugar de tratarlas como un simple sí/no.

## 🚀 Cómo ejecutarlo

```bash
# Clonar el repositorio
git clone <URL-de-tu-repo>
cd <nombre-repo>

# Instalar dependencias
pip install pyspark

# Ejecutar el notebook
jupyter notebook "Prueba - Modelo predictivo en Apache Spark MLlib.ipynb"
```

El notebook espera el archivo de datos en la ruta indicada en la variable `ruta`. Ajústala a la ubicación local del CSV antes de ejecutar (por defecto apunta a una ruta de Google Colab).

**Requisitos:** Python 3.8+, Java 8/11 (requerido por Spark), PySpark 4.x

## 📁 Estructura del repositorio

```
├── Prueba - Modelo predictivo en Apache Spark MLlib.ipynb   # Notebook principal
├── data/
│   └── ventas_simuladas.csv                                 # Dataset de entrada
└── README.md
```

## 🔭 Limitaciones y próximos pasos

- **Etiqueta sintética:** al derivarse de una regla determinística sobre `Monto_Total` y `Hora`, las métricas altas reflejan en parte que el modelo reaprende esa regla. Con datos históricos realmente etiquetados el desempeño sería más exigente y más informativo.
- **Volumen reducido:** solo 180 registros quedaron disponibles tras la limpieza, lo que limita la generalización.
- **Próximos pasos:** validación cruzada y ajuste de hiperparámetros (`numTrees`, `maxDepth`, `minInstancesPerNode`) con `CrossValidator`, selección de variables, evaluación con métricas específicas de detección de anomalías (Precision y Recall de la clase riesgosa), y persistencia del modelo entrenado para inferencia en producción.

## 👤 Autor
Harold Rodríguez B. — [LinkedIn](https://www.linkedin.com/in/harold-rodriguez-boisset/) 
