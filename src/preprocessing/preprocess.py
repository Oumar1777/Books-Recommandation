import findspark
findspark.init()
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType
from pyspark.sql.window import Window
from pyspark.sql import functions as F

def handle_missing_values(df):
    """
    Gère les valeurs manquantes dans un DataFrame Spark.
    """
    df_cleaned = df.na.drop()
    return df_cleaned

def merge_dataframes(df1, df2, on_column):
    """
    Fusionne deux DataFrames Spark sur une colonne spécifiée.
    """
    merged_df = df1.join(df2, on_column)
    return merged_df

def preprocess_ratings(rating_df, nb=100):
    """
    Prétraite le DataFrame des évaluations (ratings).
    """
    # Filtrer les utilisateurs ayant plus de nb évaluations
    if nb > 0:
        ratings_count = rating_df.groupBy('User-ID').agg(F.count('Book-Rating').alias('num_ratings'))
        rating_df = rating_df.join(ratings_count, 'User-ID', 'inner')
        rating_df = rating_df.filter(col('num_ratings') >= nb).drop('num_ratings')

    # Remplacer les 0 par la moyenne des notes du livre
    window_spec = Window().partitionBy('ISBN')
    rating_df = rating_df.withColumn('mean_rating', F.avg('Book-Rating').over(window_spec))
    rating_df = rating_df.withColumn('Book-Rating', F.when(col('Book-Rating') == 0, col('mean_rating')).otherwise(col('Book-Rating')))

    # Convertir la note en entier
    rating_df = rating_df.withColumn('Book-Rating', col('Book-Rating').cast(IntegerType()))

    # Supprimer la colonne temporaire 'mean_rating'
    rating_df = rating_df.drop('mean_rating')
    rating_df = rating_df.dropDuplicates()

    return rating_df


def preprocess_books(book_df):
    """
    Prétraite le DataFrame des livres (books).
    """
   # Supprimer les colonnes 'Image-URL-S', 'Image-URL-M', 'Image-URL-L'
    book_df = book_df.drop('Image-URL-S', 'Image-URL-M', 'Image-URL-L')
    # Supprimer les lignes avec des valeurs manquantes
    book_df = book_df.dropna()
    # Convertir la colonne 'Year-Of-Publication' en entier
    book_df = book_df.withColumn('Year-Of-Publication', col('Year-Of-Publication').cast(IntegerType()))
    # Supprimer les duplicata
    book_df = book_df.dropDuplicates()
    # Filtrer les livres avec une date de publication entre 1900 et 2022
    book_df = book_df.filter((col('Year-Of-Publication') >= 1900) & (col('Year-Of-Publication') <= 2022))
    
    return book_df

def preprocess_users(users_df):
    """
    Prétraite le DataFrame des utilisateurs (users).
    """
    users_df = users_df.drop('Age')
    
    return users_df

def preprocess(books, ratings, users,nbBooks = 50, nbRatings = 100, file_path ="data/processed/output.csv"):
    """
    Etapes de Préprocessing
    """
    books = preprocess_books(books)
    ratings = preprocess_ratings(ratings, nbRatings)
    users = preprocess_users(users)

    df = merge_dataframes(ratings, books, "ISBN")
    # Filtrer les livres ayant plus de nbBooks évalutations
    if nbBooks > 0:
        df_count = df.groupBy('Book-Title').agg(F.count('Book-Rating').alias('num_ratings'))
        df = df.join(df_count, 'Book-Title', 'inner')
        df = df.filter(col('num_ratings') >= nbBooks).drop('num_ratings')

    df = df.dropDuplicates()
    df.toPandas().to_csv(file_path, header=True)
    
    return df

def pivot(data_df):
    user_book_matrix = data_df.groupBy('Book-Title')\
        .pivot('User-ID')\
        .agg(F.first('Book-Rating'))
    user_book_matrix = user_book_matrix.na.fill(0)

    return user_book_matrix