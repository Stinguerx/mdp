from bs4 import BeautifulSoup
from pyspark.sql.functions import udf, col, year
from pyspark.sql.types import StringType


def clean_html(html):
    if html is None:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)

clean_html_udf = udf(clean_html, StringType())


def load_data(spark, folder):
    questions_df = spark.read.csv(f"./{folder}/Questions.csv", header=True, inferSchema=True)
    answers_df = spark.read.csv(f"./{folder}/Answers.csv", header=True, inferSchema=True)
    tags_df = spark.read.csv(f"./{folder}/Tags.csv", header=True, inferSchema=True)
    programming_languages = [
        "javascript", "java", "python", "c#", "php", "c++", "c", "r", "ruby", "scala", "go", "matlab"
    ]

    filtered_tags_df = tags_df.filter(col("Tag").isin(programming_languages))

    # Get the list of question IDs with programming language tags
    programming_question_ids = filtered_tags_df.select("Id").distinct()

    # Filter questions to include only those with programming language tags
    filtered_questions_df = questions_df.join(programming_question_ids, "Id", "inner")

    # Filter answers to include only those related to filtered questions
    filtered_answers_df = answers_df.join(programming_question_ids, answers_df.ParentId == programming_question_ids.Id, "inner")

    # Range of years to filter
    end_year = 2015
    start_year = 2000 

    # Filter questions and answers to exclude erroneous years
    filtered_questions_df = filtered_questions_df.filter(year(col("CreationDate")) <= end_year).filter(year(col("CreationDate")) >= start_year)
    filtered_answers_df = filtered_answers_df.filter(year(col("CreationDate")) <= end_year).filter(year(col("CreationDate")) >= start_year)

    # Clean HTML from 'Body' column
    clean_questions_df = filtered_questions_df.withColumn("CleanBody", clean_html_udf(filtered_questions_df.Body))
    clean_answers_df = filtered_answers_df.withColumn("CleanBody", clean_html_udf(filtered_answers_df.Body))

    return clean_questions_df, clean_answers_df, filtered_tags_df