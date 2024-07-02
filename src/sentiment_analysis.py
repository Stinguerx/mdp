from pyspark.sql.functions import udf, col, year, when, sum
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
from pyspark.sql.types import FloatType

custom_words = {
    'bug': -2.0,
    'error': -1.5,
    'crash': -2.5,
    'feature': 1.5,
    'efficient': 2.0,
    'slow': -1.5,
    'fast': 1.5,
    'optimized': 2.0,
    'clean': 1.5,
    'messy': -1.5,
    'intuitive': 2.0,
    'confusing': -2.0,
    'powerful': 2.5,
    'limited': -1.5,
}

# Update VADER lexicon
sia.lexicon.update(custom_words)


@udf(FloatType())
def get_sentiment(text):
    return sia.polarity_scores(text)['compound']


def analyze_language_sentiment(questions_df, answers_df, tags_df):
    combined_df = questions_df.select("Id", "CreationDate", "CleanBody") \
        .union(answers_df.select("ParentId", "CreationDate", "CleanBody"))
    
    combined_df = combined_df.join(tags_df, combined_df.Id == tags_df.Id, "left")
    combined_df = combined_df.withColumn("Sentiment", get_sentiment("CleanBody"))
    
    sentiment_by_year = combined_df.groupBy(year("CreationDate").alias("Year"), "Tag") \
        .agg({"Sentiment": "avg"}) \
        .withColumnRenamed("avg(Sentiment)", "AvgSentiment") \
        .orderBy("Year")
    
    sentiment_by_year.show(100, truncate=False)
    
    return sentiment_by_year


def analyze_user_expertise(questions_df, answers_df, tags_df):
    # Combine user activities from questions and answers
    user_activities = questions_df.select("OwnerUserId", "Score").union(
        answers_df.select("OwnerUserId", "Score")
    )

    # Calculate total score for each user
    user_scores = user_activities.groupBy("OwnerUserId").agg(sum("Score").alias("TotalScore"))
    
    # Categorize users based on their total score
    users_with_expertise = user_scores.withColumn(
        "ExpertiseLevel",
        when(col("TotalScore") < 50, "Novice")
        .when((col("TotalScore") >= 50) & (col("TotalScore") < 200), "Beginner")
        .when((col("TotalScore") >= 200) & (col("TotalScore") < 500), "Intermediate")
        .when((col("TotalScore") >= 500) & (col("TotalScore") < 1000), "Advanced")
        .otherwise("Expert")
    )
    
    # Join with questions to analyze sentiment by expertise level
    questions_with_tags = questions_df.join(tags_df, questions_df.Id == tags_df.Id, "inner")
    questions_with_expertise = questions_with_tags.join(users_with_expertise, "OwnerUserId")
    questions_with_sentiment = questions_with_expertise.withColumn("Sentiment", get_sentiment("CleanBody"))

    expertise_order = when(col("ExpertiseLevel") == "Novice", 1) \
                      .when(col("ExpertiseLevel") == "Beginner", 2) \
                      .when(col("ExpertiseLevel") == "Intermediate", 3) \
                      .when(col("ExpertiseLevel") == "Advanced", 4) \
                      .when(col("ExpertiseLevel") == "Expert", 5)

    sentiment_by_expertise = questions_with_sentiment.groupBy("ExpertiseLevel", "Tag") \
        .agg({"Sentiment": "avg"}) \
        .withColumnRenamed("avg(Sentiment)", "AvgSentiment") \
        .orderBy("Tag", expertise_order)
    
    return sentiment_by_expertise