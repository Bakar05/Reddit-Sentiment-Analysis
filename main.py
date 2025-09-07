from api_key import reddit
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from database_credentials import engine
import warnings

warnings.filterwarnings('ignore')
nltk.download('vader_lexicon')


def log_progress(message):
    '''
    This function logs all major steps of the pipeline.
    Helps track workflow and debug errors by writing messages to a log file.
    '''
    time_stamp_format = '%Y-%b-%d-%H:%M:%S'
    now = datetime.now()
    time_stamp = now.strftime(time_stamp_format)

    with open("code_log.txt", "a") as f:
        f.write(time_stamp + ' : ' + message + '\n')


def take_topic():
    '''
    Takes topic input from the user, runs ETL + sentiment analysis pipeline.
    Saves cleaned and analyzed data to CSV files.
    Returns: topic and sentiment dataframe.
    '''
    while True:
        topic = input("Enter a topic: ").strip()

        if not topic:
            print("Topic cannot be empty. Try again.")
            continue

        file_name = f"reddit_{topic}_data.csv"
        file_name_sentiment = f"sentimental_analysis_{topic}_data.csv"

        log_progress(f"Starting analysis for topic: {topic}")

        print("Extracting data from Reddit...")
        raw_df = extract(topic)

        if raw_df.empty:
            print(f"No data extracted for '{topic}'. Please try another topic.")
            continue

        print("Cleaning and preprocessing data...")
        clean_df = transform(raw_df)

        load_to_csv(clean_df, file_name)
        print("Performing sentiment analysis...")

        sentimental_analysis_df = perform_sentimental_analysis(clean_df)
        load_to_csv(sentimental_analysis_df, file_name_sentiment)

        load_to_db(sentimental_analysis_df, table_name="reddit_posts")

        return topic, sentimental_analysis_df


def extract(topic):
    '''
    Extracts Reddit posts for a given topic from r/all.
    Returns a DataFrame with basic metadata: title, text, subreddit, score, comments, etc.
    '''
    try:
        subreddit = reddit.subreddit("all")
        posts = subreddit.search(topic, limit=500)

        data = []
        for post in posts:
            data.append([
                post.created_utc, post.title,
                post.selftext, post.subreddit.display_name,
                post.score, post.num_comments,
                post.author.name if post.author else "[deleted]",
                post.url
            ])

        df = pd.DataFrame(data,
                          columns=["timestamp", "Title", "Text", "subreddit", "score", "comments", "author", "url"])

        df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime('%Y-%b-%d-%H:%M:%S')
        log_progress(f"Data extraction complete for topic: {topic}. Found {len(df)} posts.")
        return df

    except Exception as e:
        log_progress(f"Error extracting topic:{topic}, {str(e)}")
        return pd.DataFrame()


def transform(df):
    '''
    Cleans Reddit text data by removing:
    - URLs
    - Special characters
    - Extra whitespaces
    - Short/empty texts

    Returns cleaned dataframe.
    '''
    try:
        clean_df = df.copy()
        url_pattern = r'http\S+|www.\S+'
        special_chars = r'[^a-zA-Z\s]'
        more_then_one_whitespaces = r'\s+'

        clean_df["Text"] = clean_df["Text"].str.replace(url_pattern, '', regex=True)
        clean_df["Text"] = clean_df["Text"].str.replace(special_chars, '', regex=True)
        clean_df["Text"] = clean_df["Text"].str.replace(more_then_one_whitespaces, ' ', regex=True)
        clean_df["Text"] = clean_df["Text"].str.lower().str.strip()

        clean_df["Text"] = clean_df["Text"].replace(r'^\s*$', pd.NA, regex=True)
        clean_df = clean_df.dropna(subset=["Text"])
        clean_df = clean_df[clean_df["Text"].str.len() > 10]

        clean_df = clean_df.sort_values(by="date").reset_index(drop=True)
        clean_df = clean_df.drop(columns=["timestamp"], errors='ignore')

        log_progress("Data Preprocessing complete.")
        return clean_df

    except Exception as e:
        log_progress(f"Error cleaning Text: {str(e)}")
        return df


def perform_sentimental_analysis(df):
    '''
    Performs sentiment analysis using NLTK’s VADER model.
    Adds sentiment scores (positive, negative, neutral, compound).
    Returns dataframe with new sentiment columns.
    '''
    try:
        sia = SentimentIntensityAnalyzer()
        sentiment_df = df.copy()

        sentiment_scores = sentiment_df["Text"].apply(sia.polarity_scores)
        sentiment_df = pd.concat([sentiment_df, sentiment_scores.apply(pd.Series)], axis=1)

        sentiment_df = sentiment_df.rename(columns={
            "neg": "negative_score",
            "neu": "neutral_score",
            "pos": "positive_score",
            "compound": "final_rating"
        })

        log_progress("Sentiment Analysis complete.")
        return sentiment_df

    except Exception as e:
        log_progress(f"Error performing Sentiment Analysis, {str(e)}")
        return df


def plot_histogram(df):
    ''' Plots histogram of sentiment scores across posts. '''
    try:
        if df.empty:
            print("No data to analyze.")
            return df

        plt.figure(figsize=(12, 7))
        sns.histplot(df['final_rating'], bins=20, kde=False, color="#2E86AB")

        plt.title("Sentiment Score Distribution", fontsize=16, fontweight='bold')
        plt.xlabel("Sentiment Score")
        plt.ylabel("Frequency")

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig("sentiment_score_distribution.png")

        plt.show()

        log_progress('Histogram chart successfully plotted.')

    except Exception as e:
        log_progress(f"Error plotting Histogram chart: {str(e)}")


def plot_subreddit_distribution(df, topic):
    ''' Shows top subreddits where the topic is discussed. '''
    try:
        if df.empty:
            print("No data to analyze.")
            return df

        plt.figure(figsize=(14, 8))
        top_subreddits = df['subreddit'].value_counts().head(10)

        colors = sns.color_palette('Set2', len(top_subreddits))
        plt.bar(range(len(top_subreddits)), top_subreddits.values, color=colors, alpha=0.8)

        plt.title(f"Top Subreddits Discussing '{topic}'", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Subreddit")
        plt.ylabel("Number of Posts")

        plt.xticks(range(len(top_subreddits)), top_subreddits.index, rotation=45, ha='right')

        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        safe_topic = topic.replace(" ", "_")
        plt.savefig(f"subreddit_distribution_{safe_topic}.png")

        plt.show()

        log_progress('Subreddit distribution chart plotted with vertical bars.')

    except Exception as e:
        log_progress(f"Error plotting subreddit distribution: {str(e)}")


def plot_engagement_metrics(df, topic):
    ''' Shows distributions of upvotes and comments for posts about the topic. '''
    try:
        if df.empty:
            print("No data to analyze.")
            return df

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        sns.histplot(df['score'], bins=20, ax=ax1, color='#2E86AB')

        ax1.set_title(f"Upvote Distribution for '{topic}'")
        ax1.set_xlabel("Upvotes")
        ax1.set_ylabel("Frequency")
        ax1.grid(True, alpha=0.3)

        sns.histplot(df['comments'], bins=20, ax=ax2, color='#A23B72')

        ax2.set_title(f"Comments Distribution for '{topic}'")
        ax2.set_xlabel("Number of Comments")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        safe_topic = topic.replace(" ", "_")
        plt.savefig(f"engagement_metrics_{safe_topic}.png")

        plt.show()

        log_progress('Engagement metrics chart plotted.')

    except Exception as e:
        log_progress(f"Error plotting engagement metrics: {str(e)}")


def plot_sentiment_vs_engagement(df, topic):
    '''
    Scatterplots:
    - Sentiment vs Upvotes
    - Sentiment vs Comments
    Also calculates correlation values.
    '''
    try:
        if df.empty:
            print("No data to analyze.")
            return df

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        sns.scatterplot(data=df, x='final_rating', y='score', alpha=0.6, ax=ax1)

        ax1.set_title(f"Sentiment vs Upvotes for '{topic}'")
        ax1.set_xlabel("Sentiment Score")
        ax1.set_ylabel("Upvotes")
        ax1.grid(True, alpha=0.3)

        sns.scatterplot(data=df, x='final_rating', y='comments', alpha=0.6, ax=ax2)

        ax2.set_title(f"Sentiment vs Comments for '{topic}'")
        ax2.set_xlabel("Sentiment Score")
        ax2.set_ylabel("Comments")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        safe_topic = topic.replace(" ", "_")
        plt.savefig(f"sentiment_engagement_{safe_topic}.png")

        plt.show()

        corr_upvotes = df['final_rating'].corr(df['score'])
        corr_comments = df['final_rating'].corr(df['comments'])
        print(f"Correlation between sentiment and upvotes: {corr_upvotes:.3f}")
        print(f"Correlation between sentiment and comments: {corr_comments:.3f}")

        log_progress('Sentiment vs engagement chart plotted.')
    except Exception as e:
        log_progress(f"Error plotting sentiment vs engagement: {str(e)}")


def sentiment_by_subreddit(df, topic):
    '''
    Shows average sentiment per subreddit.
    Only includes subreddits with >= 3 posts.
    '''
    try:
        if df.empty:
            print("No data to analyze.")
            return df

        sentiment_by_sub = df.groupby('subreddit')['final_rating'].agg(['mean', 'count']).round(3)
        sentiment_by_sub = sentiment_by_sub[sentiment_by_sub['count'] >= 3]

        plt.figure(figsize=(12, 8))
        colors = ['red' if x < 0 else 'green' for x in sentiment_by_sub['mean']]

        plt.barh(sentiment_by_sub.index, sentiment_by_sub['mean'], color=colors, alpha=0.7)
        plt.title(f"Average Sentiment by Subreddit for '{topic}'", fontsize=16)
        plt.xlabel("Average Sentiment Score")

        plt.tight_layout()

        safe_topic = topic.replace(" ", "_")
        plt.savefig(f"sentiment_by_subreddit_{safe_topic}.png")

        plt.show()
    except Exception as e:
        log_progress(f"Error plotting sentiment by subreddit: {str(e)}")


def compare_topics(df1_sentiment, topic1):
    '''
    Asks for 2 topics, compares their average sentiment side by side.
    Plots a comparison bar chart.
    '''
    try:
        if df1_sentiment.empty:
            print("No data to analyze.")
            return df1_sentiment

        topic2 = input("Enter second topic: ")

        df1_sentiment = df1_sentiment.copy()
        df1_sentiment['topic'] = topic1

        df2 = transform(extract(topic2))
        if df2.empty:
            print(f"No data found for {topic2}")
            return

        df2_sentiment = perform_sentimental_analysis(df2)
        df2_sentiment['topic'] = topic2

        combined_df = pd.concat([df1_sentiment, df2_sentiment])
        comparison = combined_df.groupby('topic')['final_rating'].mean().reset_index()

        plt.figure(figsize=(10, 6))
        sns.barplot(data=comparison, x='topic', y='final_rating', palette="viridis")

        plt.title("Average Sentiment Comparison Between Two Topics", fontsize=16, fontweight='bold')
        plt.xlabel("Topic")
        plt.ylabel("Average Sentiment Score")

        plt.ylim(-1, 1)

        plt.xticks(rotation=30, ha='right')

        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        safe_topic1 = topic1.replace(" ", "_")
        safe_topic2 = topic2.replace(" ", "_")

        plt.savefig(f"Comparison_between_{safe_topic1}_and_{safe_topic2}.png")
        plt.show()

        log_progress("Comparison chart for two topics successfully plotted.")

    except Exception as e:
        log_progress(f"Error in comparing two topics: {str(e)}")


def load_to_csv(df, output_path):
    ''' Saves dataframe to CSV and logs the action. '''
    try:
        df.to_csv(output_path, index=False)
        log_progress(f'Data saved to {output_path}')
    except Exception as e:
        log_progress(f"Error saving to CSV: {str(e)}")


def load_to_db(df, table_name="reddit_posts"):
    """Loads a dataframe into MySQL table."""
    try:
        df.to_sql(table_name, con=engine, if_exists="append", index=False)

        log_progress(f"Data saved to MySQL table {table_name}")
        print(f"Data saved to MySQL table {table_name}")

    except Exception as e:
        log_progress(f"Error saving to MySQL: {str(e)}")

        print(f"Error saving to MySQL: {str(e)}")


def main():
    topic, sentimental_analysis_df = take_topic()

    if not topic or sentimental_analysis_df is None or sentimental_analysis_df.empty:
        print("No valid data to analyze. Exiting...")
        return

    while True:
        print(
            "CHOOSE AN ANALYSIS:\n"
            "1:  Sentiment distribution (Histogram)\n"
            "2:  Subreddit distribution - Where topic is discussed\n"
            "3:  Engagement metrics - Upvotes and comments analysis\n"
            "4:  Sentiment vs Engagement correlation\n"
            "5:  Average sentiment by subreddit\n"
            "6:  Compare with another topic\n"
            "7:  Change the topic\n"
            "8:  Exit"
        )
        choice = input("Enter a number (1-8): ")

        if choice == "1":
            plot_histogram(sentimental_analysis_df)

        elif choice == "2":
            plot_subreddit_distribution(sentimental_analysis_df, topic)

        elif choice == "3":
            plot_engagement_metrics(sentimental_analysis_df, topic)

        elif choice == "4":
            plot_sentiment_vs_engagement(sentimental_analysis_df, topic)

        elif choice == "5":
            sentiment_by_subreddit(sentimental_analysis_df, topic)

        elif choice == "6":
            compare_topics(sentimental_analysis_df, topic)

        elif choice == "7":
            topic, sentimental_analysis_df = take_topic()

            if not topic or sentimental_analysis_df is None or sentimental_analysis_df.empty:
                print("No valid data to analyze. Exiting...")
                return

        elif choice == "8":
            print("Exiting...")

            log_progress(f"Analysis completed for topic: {topic}")
            break

        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()


