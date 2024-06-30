from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, year

# Create a Spark session
spark = SparkSession.builder.appName("proyecto_patos").getOrCreate()

# Read the dataset
questions_df = spark.read.csv("data/Questions.csv", header=True)
answers_df = spark.read.csv("data/Answers.csv", header=True)
tags_df = spark.read.csv("data/Tags.csv", header=True)

# Join id questions y tags
data1 = questions_df.join(tags_df, questions_df.Id == tags_df.Id, "inner")

# Extraer año de creacion
data1 = data1.withColumn("Year", year(col("CreationDate")))

spark.read.option('header', 'true').csv('data/Questions.csv').show(10)
data1.show(10)

# Contar popularidad por año
popularidad_anual = data1.groupBy("Year", "Tag").agg(count("Tag").alias("conteo")).orderBy(col("Year"), col("conteo").desc())
popularidad_anual.show()

# Stop the Spark session
spark.stop()
