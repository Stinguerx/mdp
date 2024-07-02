import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pyspark.sql.functions import collect_list
import os

def plot_language_popularity(popularity_results):
    pdf = popularity_results.toPandas()
    
    tags = pdf['Tag'].unique()
    colormap = plt.get_cmap('tab20', len(tags))  # 'tab20' is a colormap with 20 distinct colors

    plt.figure(figsize=(15, 8))

    # Plot each language with a unique color from the colormap
    for i, tag in enumerate(tags):
        lang_data = pdf[pdf['Tag'] == tag]
        plt.plot(lang_data['Year'], lang_data['Count'], label=tag, color=colormap(i))
    
    plt.title('Language Popularity over the Years')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('output/language_popularity_trend.png')
    plt.close()


def plot_language_sentiment(sentiment_results):
    pdf = sentiment_results.toPandas()
    
    tags = pdf['Tag'].unique()
    colormap = plt.get_cmap('tab20', len(tags))  # 'tab20' is a colormap with 20 distinct colors

    plt.figure(figsize=(15, 8))

    # Plot each language with a unique color from the colormap
    for i, tag in enumerate(tags):
        lang_data = pdf[pdf['Tag'] == tag]
        plt.plot(lang_data['Year'], lang_data['AvgSentiment'], label=tag, color=colormap(i))

    # plt.figure(figsize=(15, 8))
    # for lang in pdf['Tag'].unique():
    #     lang_data = pdf[pdf['Tag'] == lang]
    #     plt.plot(lang_data['Year'], lang_data['AvgSentiment'], label=lang)
    
    plt.title('Language Sentiment over the Years')
    plt.xlabel('Year')
    plt.ylabel('Average Compound Sentiment')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('output/language_sentiment_trend.png')
    plt.close()


def plot_popularity_sentiment_correlation(popularity_results, sentiment_results):
    pop_df = popularity_results.select("Year", "Tag", "Count")
    sent_df = sentiment_results.select("Year", "Tag", "AvgSentiment")
    
    corr_df = pop_df.join(sent_df, ["Year", "Tag"])
    corr_pd = corr_df.toPandas()
    
    correlation = corr_pd['Count'].corr(corr_pd['AvgSentiment'])
    
    plt.figure(figsize=(10, 6))
    plt.scatter(corr_pd['Count'], corr_pd['AvgSentiment'])
    plt.title(f'Correlation between Language Popularity and Sentiment\nCorrelation: {correlation:.2f}')
    plt.xlabel('Popularity (Count)')
    plt.ylabel('Average Compound Sentiment')
    plt.savefig('output/popularity_sentiment_correlation.png')
    plt.close()


def plot_expertise_sentiment(expertise_results):
    pdf = expertise_results.toPandas()

    # Get unique tags and generate a color map
    tags = pdf['Tag'].unique()
    num_tags = len(tags)
    color_map = plt.get_cmap('tab20', num_tags)  # Choose a colormap and number of colors

    plt.figure(figsize=(15, 8))

    for i, tag in enumerate(tags):
        tag_data = pdf[pdf['Tag'] == tag]
        color = color_map(i / num_tags)  # Get color from colormap
        plt.plot(tag_data['ExpertiseLevel'], tag_data['AvgSentiment'], marker='o', label=tag, color=color)

    plt.title('Sentiment by Expertise Level and Language')
    plt.xlabel('Expertise Level')
    plt.ylabel('Average Sentiment')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('output/expertise_sentiment.png')
    plt.close()


def generate_word_clouds(questions_df, tags_df):
    questions_with_tags = questions_df.join(tags_df, questions_df.Id == tags_df.Id, "inner")
    words_by_lang = questions_with_tags.groupBy("Tag").agg(collect_list("CleanBody").alias("texts"))
    words_by_lang_pd = words_by_lang.toPandas()
    
    os.makedirs('output/wordclouds', exist_ok=True)

    for _, row in words_by_lang_pd.iterrows():
        lang = row['Tag']
        text = ' '.join(row['texts'])
        
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {lang}')
        plt.savefig(f'output/wordclouds/wordcloud_{lang}.png')
        plt.close()
