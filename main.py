from pyspark.sql import SparkSession
from src.data_loading import load_data
from src.popularity_analysis import analyze_language_popularity
from src.sentiment_analysis import analyze_language_sentiment, analyze_user_expertise
from src.visualization import (
    plot_language_popularity,
    plot_language_sentiment,
    plot_popularity_sentiment_correlation,
    plot_expertise_sentiment,
    generate_word_clouds,
)

def main():
    # Initialize Spark session
    spark = SparkSession.builder.appName("StackOverflowAnalysis").config("spark.driver.memory", "8g").config("spark.executor.memory", "8g").getOrCreate()

    # Load data
    questions_df, answers_df, tags_df = load_data(spark, "data")

    # Perform analyses

    # Language popularity and sentiment
    popularity_results = analyze_language_popularity(questions_df, tags_df)
    plot_language_popularity(popularity_results)

    sentiment_results = analyze_language_sentiment(questions_df, answers_df, tags_df)
    plot_language_sentiment(sentiment_results)

    plot_popularity_sentiment_correlation(popularity_results, sentiment_results)

    

    # Sentiment by user expertise and language
    expertise_results = analyze_user_expertise(questions_df, answers_df, tags_df)
    plot_expertise_sentiment(expertise_results)

    # Word clouds
    generate_word_clouds(questions_df, tags_df)

    # Topic modeling
    # topic_results, vocab = perform_topic_modeling(questions_df)
    # plot_topic_distribution(topic_results, vocab)

    # Language co-occurrence
    # cooccurrence_results = analyze_language_cooccurrence(questions_df)
    # plot_language_cooccurrence_network(cooccurrence_results)

    spark.stop()

if __name__ == "__main__":
    main()