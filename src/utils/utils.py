import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, sum, countDistinct, count

def create_spark_session(app_name="RecommendationApp"):
    """
    Crée une session Spark.
    """
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    return spark

def load_data(spark, file_path, format="csv", header=True, infer_schema=True):
    """
    Charge les données à partir d'un fichier dans un DataFrame Spark.
    """
    df = spark.read.format(format).option("header", header).option("inferSchema", infer_schema).load(file_path)
    return df

def count_missing_values(spark_df):
    """
    Compte le nombre de valeurs nulls ou nans dans chaque colonne
    """
    expressions = [
        sum(when(col(c).isNull(), 1).otherwise(0)) 
        .alias(c) for c in spark_df.columns
    ]
    
    result_df = spark_df.agg(*expressions)
    
    return result_df

def count_distinct_values(spark_df):
    """
    Compte le nombre de valeurs uniques dans chaque colonne
    """
    expressions = [
        countDistinct(col(c)).alias(f"{c}")
        for c in spark_df.columns
    ]
    
    result_df = spark_df.agg(*expressions)
    
    return result_df
