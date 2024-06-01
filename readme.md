# Proyecto de Clustering con K-Means y DBSCAN

## Yasmin Alejandra Giraldo Rendón
## Jhorman David Rodriguez Rios

Este proyecto utiliza técnicas de clustering para analizar el dataset de Iris. Se emplean los algoritmos K-Means y DBSCAN para agrupar los datos y se evalúan los resultados utilizando el coeficiente de Silhouette.

## Bibliotecas de Python Utilizadas

- **pandas**: Para manipulación y análisis de datos.
- **numpy**: Para cálculos numéricos.
- **scikit-learn**: Para algoritmos de aprendizaje automático y preprocesamiento de datos.
  - `KMeans`
  - `DBSCAN`
  - `StandardScaler`
  - `silhouette_score`
- **matplotlib**: Para visualización de datos.
- **seaborn**: Para visualización de datos y carga del dataset Iris.

## Estructura del Proyecto

- **src/clustering.py**: Contiene las funciones para aplicar K-Means y DBSCAN, y para plotear y guardar las gráficas de los clusters.
- **src/data_processing.py**: Contiene las funciones para cargar y procesar los datos.
- **notebooks**: Contiene los Jupyter Notebooks con el análisis y visualización de los datos.

## Uso

1. **Cargar y procesar los datos**:
    ```python
    from data_processing import load_data, process_data

    data_path = '../data/raw/iris.csv'
    df = load_data(data_path)
    df_scaled = process_data(df.drop(columns=['species']))
    ```

2. **Aplicar K-Means y plotear los clusters**:
    ```python
    from clustering import apply_kmeans, plot_clusters, save_kmeans_plot

    km, y_km = apply_kmeans(df_scaled, n_clusters=3)
    fig_kmeans = plot_clusters(df_scaled, y_km, km)
    save_kmeans_plot(fig_kmeans, 'kmeans_clusters.png')
    ```

3. **Calcular el coeficiente de Silhouette para K-Means**:
    ```python
    from sklearn.metrics import silhouette_score

    silhouette_avg_kmeans = silhouette_score(df_scaled, y_km)
    print(f'Silhouette Score for K-Means: {silhouette_avg_kmeans}')
    ```

4. **Aplicar DBSCAN y plotear los clusters**:
    ```python
    from clustering import apply_dbscan, plot_dbscan, save_dbscan_plot

    db, y_db = apply_dbscan(df_scaled, eps=0.2, min_samples=5)
    fig_dbscan = plot_dbscan(df_scaled, y_db)
    save_dbscan_plot(fig_dbscan, 'dbscan_clusters.png')
    ```

5. **Calcular el coeficiente de Silhouette para DBSCAN**:
    ```python
    if len(set(y_db)) > 1:
        silhouette_avg_dbscan = silhouette_score(df_scaled, y_db)
        print(f'Silhouette Score for DBSCAN: {silhouette_avg_dbscan}')
    else:
        print('DBSCAN no encontró suficientes clusters distintos para calcular el silhouette score')
    ```

## Resultados

### K-Means
- Número de Clusters: 3
- Silhouette Score: 0.55
- Gráfico de Clusters: `reports/figures/kmeans_clusters.png`

### DBSCAN
- Parámetros: `eps=0.2`, `min_samples=5`
- Silhouette Score: 0.48 (si se encuentran suficientes clusters)
- Gráfico de Clusters: `reports/figures/dbscan_clusters.png`

## Estructuras de Datos Utilizadas

Durante el proceso se utilizaron las siguientes estructuras de datos de Python:
- **`DataFrames` de `pandas`**: Para manipulación y análisis de datos.
- **`Arrays` de `numpy`**: Para cálculos numéricos y preprocesamiento de datos.

## Tensores Utilizados

Para este proyecto no se utilizaron tensores explícitamente, ya que las bibliotecas utilizadas (`scikit-learn`, `pandas`, `numpy`) trabajan principalmente con arrays y dataframes. Los tensores son más comunes en proyectos que utilizan bibliotecas como TensorFlow o PyTorch para aprendizaje profundo.

## Conclusiones

- **K-Means**:
  - Es un algoritmo de clustering simple y eficiente.
  - Requiere que se especifique el número de clusters de antemano.
  - El coeficiente de Silhouette obtenido puede ayudar a determinar la calidad del clustering.

- **DBSCAN**:
  - No requiere que se especifique el número de clusters de antemano.
  - Puede identificar clusters de forma arbitraria y detectar ruido.
  - La elección de los parámetros `eps` y `min_samples` es crucial para obtener buenos resultados.
  - El coeficiente de Silhouette puede no ser calculable si no se encuentran suficientes clusters distintos.

Este proyecto demuestra la aplicación práctica de algoritmos de clustering y la importancia de evaluar sus resultados para obtener insights valiosos a partir de los datos.

