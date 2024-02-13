from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeansModel

def Using_Kmeans(datas, k = 5, feature_cols = ["Year-Of-Publication", "ISBN", "Book-Rating", "AuthorIndex", "PublisherIndex"], path="KMeansModel"):
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    kmeans_data = assembler.transform(datas)
    
    selected_cols = ["User-ID", "features"]
    kmeans_data = kmeans_data.select(selected_cols)

    kmeans = KMeans(k=k, seed=1)
    model = kmeans.fit(kmeans_data)

    predictions = model.transform(kmeans_data)
    
    model.write().overwrite().save(path)
    return model, predictions


def load_Kmeans(spark, model_path):
    """
    Charge un modèle ALS préalablement sauvegardé.

    Args:
    - model_path (str): Le chemin vers le répertoire où le modèle ALS est sauvegardé.

    Returns:
    - ALSModel: Le modèle ALS chargé.
    """
    try:
        # Chargez le modèle ALS
        loaded_model = KMeansModel.load(model_path)
        return loaded_model
    except Exception as e:
        # En cas d'erreur, imprimez un message d'erreur et fermez la session Spark
        print(f"Erreur lors du chargement du modèle ALS : {str(e)}")