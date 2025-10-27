import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from textblob import TextBlob
from datetime import datetime

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove non-ascii characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    return ' '.join(tokens)

# Read and process tweets
df = pd.read_csv('tweets.csv')
df['cleaned_content'] = df['content'].apply(clean_text)
# Fix date parsing by specifying format and setting dayfirst=True
df['date_time'] = pd.to_datetime(df['date_time'], format='%d/%m/%Y %H:%M', dayfirst=True)

# Combine all cleaned tweets into one string
all_text = ' '.join(df['cleaned_content'].dropna())

# 1. Word Frequency Bar Chart
plt.figure(figsize=(12, 6))
words, counts = zip(*Counter(all_text.split()).most_common(10))
sns.barplot(x=list(words), y=list(counts))
plt.title('Top 10 Most Common Words')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Word Cloud
plt.figure(figsize=(10, 10))
wordcloud = WordCloud(width=800, height=800,
                     background_color='white',
                     min_font_size=10).generate(all_text)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word Cloud of Tweets")
plt.tight_layout(pad=0)
plt.show()

# 3. Sentiment Analysis
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['sentiment'] = df['cleaned_content'].apply(get_sentiment)

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='sentiment', bins=30, kde=True)
plt.title('Distribution of Tweet Sentiments')
plt.xlabel('Sentiment Score (Negative -> Positive)')
plt.ylabel('Count')
plt.show()

# 4. Tweet Activity Over Time
plt.figure(figsize=(12, 6))
df.set_index('date_time').resample('D')['content'].count().plot()
plt.title('Tweet Activity Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Tweets')
plt.tight_layout()
plt.show()

# 5. Engagement Analysis
plt.figure(figsize=(10, 6))
engagement_metrics = df[['number_of_likes', 'number_of_shares']].mean()
sns.barplot(x=engagement_metrics.index, y=engagement_metrics.values)
plt.title('Average Engagement Metrics')
plt.ylabel('Count')
plt.show()

# Print summary statistics
print("\nDataset Summary:")
print(f"Total Tweets: {len(df)}")
print(f"Date Range: {df['date_time'].min().date()} to {df['date_time'].max().date()}")
print(f"Average Likes: {df['number_of_likes'].mean():.2f}")
print(f"Average Shares: {df['number_of_shares'].mean():.2f}")