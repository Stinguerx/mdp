from pyspark.sql.functions import col, count, year


def analyze_language_popularity(questions_df, tags_df):
    # Join questions and tags
    questions_with_tags = questions_df.join(tags_df, questions_df.Id == tags_df.Id, "inner")

    # Extract year of creation
    questions_with_tags = questions_with_tags.withColumn("Year", year(col("CreationDate")))

    # Calculate popularity
    popularity_by_year = questions_with_tags.groupBy("Year", "Tag") \
        .agg(count("Tag").alias("Count")) \
        .orderBy(col("Year"), col("Count").desc())

    return popularity_by_year