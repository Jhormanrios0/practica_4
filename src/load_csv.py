import seaborn as sns

# Cargar el dataset de Iris con seaborn
df = sns.load_dataset('iris')
df.to_csv('data/raw/iris.csv', index=False)
