from pyspark.ml.recommendation import ALS
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.feature import VectorAssembler

def Using_ALS(datas, maxIter = 5, feature_cols = ["Year-Of-Publication", "ISBN", "Book-Rating", "AuthorIndex", "PublisherIndex"], path="ALSModel"):
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    datas = assembler.transform(datas)

    train, test = datas.randomSplit([0.8, 0.2], seed=42)

    als = ALS(maxIter=10, regParam=0.1, userCol="User-ID", itemCol="ISBN", ratingCol="Book-Rating", coldStartStrategy="drop", implicitPrefs=False)
    model = als.fit(train)

    predictions = model.transform(test)
    
    model.write().overwrite().save(path)
    return model, predictions


def load_ALS(spark, model_path):
    """
    Charge un modèle ALS préalablement sauvegardé.

    Args:
    - model_path (str): Le chemin vers le répertoire où le modèle ALS est sauvegardé.

    Returns:
    - ALSModel: Le modèle ALS chargé.
    """
    try:
        # Chargez le modèle ALS
        loaded_model = ALSModel.load(model_path)
        return loaded_model
    except Exception as e:
        # En cas d'erreur, imprimez un message d'erreur et fermez la session Spark
        print(f"Erreur lors du chargement du modèle ALS : {str(e)}")

