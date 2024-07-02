import pandas as pd
import os
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


# Test of the main file with a subsample of the dataset for debugging purposes
# This will create the same files in the output directory as the main file
def main():

    # Subsampling, only done once
    if not os.path.exists('./data-test'):
        # Load original CSV files
        answers = pd.read_csv('data/Answers.csv', encoding='latin-1')
        questions = pd.read_csv('data/Questions.csv', encoding='latin-1')
        tags = pd.read_csv('data/Tags.csv', encoding='latin-1')

        # Merge the dataframes on the 'Id' column
        merged_df = answers.merge(questions, left_on='ParentId', right_on='Id', suffixes=('', '_question'))
        merged_df = merged_df.merge(tags, left_on='ParentId', right_on='Id', suffixes=('', '_tag'))

        # Subsample the merged dataframe
        subsampled_df = merged_df.sample(n=10000, random_state=42)

        # Save the subsample to new CSV files with original headers
        subsampled_answers = subsampled_df[['Id', 'OwnerUserId', 'CreationDate', 'ParentId', 'Score', 'Body']]
        subsampled_questions = subsampled_df[['Id_question', 'OwnerUserId_question', 'CreationDate_question', 'ClosedDate', 'Score_question', 'Title', 'Body_question']].rename(columns={
            'Id_question': 'Id',
            'OwnerUserId_question': 'OwnerUserId',
            'CreationDate_question': 'CreationDate',
            'Score_question': 'Score',
            'Body_question': 'Body'
        })
        subsampled_tags = subsampled_df[['Id_tag', 'Tag']].rename(columns={'Id_tag': 'Id'})

        os.makedirs('./data-test')
        subsampled_answers.to_csv('data-test/Answers.csv', index=False)
        subsampled_questions.to_csv('data-test/Questions.csv', index=False)
        subsampled_tags.to_csv('data-test/Tags.csv', index=False)

        print("Subsampled CSV files created successfully.")

    # Actual test begins here
    spark = SparkSession.builder.appName("StackOverflowAnalysis").config("spark.driver.memory", "8g").config("spark.executor.memory", "8g").getOrCreate()
    questions_df, answers_df, tags_df = load_data(spark, "data")

    # Perform analyses
    # Language popularity and sentiment
    popularity_results = analyze_language_popularity(questions_df, tags_df)
    plot_language_popularity(popularity_results)

    sentiment_results = analyze_language_sentiment(questions_df, answers_df, tags_df)
    plot_language_sentiment(sentiment_results)

    plot_popularity_sentiment_correlation(popularity_results, sentiment_results)

    #Most Popular language by year
    

    # Sentiment by user expertise and language
    expertise_results = analyze_user_expertise(questions_df, answers_df, tags_df)
    plot_expertise_sentiment(expertise_results)

    # Word clouds
    generate_word_clouds(questions_df, tags_df)

    spark.stop()


if __name__ == "__main__":
    main()