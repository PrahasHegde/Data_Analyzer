# Data Analyzer
# Twitter Data Analysis Tool

A Python-based tool for analyzing Twitter data and generating insightful visualizations.

## Features

- Text cleaning and preprocessing
- Word frequency analysis
- Word cloud generation
- Sentiment analysis
- Time series analysis of tweet activity
- Engagement metrics visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/twitter-analysis-tool.git
cd twitter-analysis-tool
```

2. Install required packages:
```bash
pip install pandas nltk wordcloud textblob seaborn matplotlib
```

3. Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Usage

1. Prepare your Twitter data in CSV format with the following columns:
   - content: Tweet text
   - date_time: Tweet timestamp (format: DD/MM/YYYY HH:MM)
   - number_of_likes: Number of likes
   - number_of_shares: Number of shares

2. Run the analysis:
```bash
python text_analyzer.py
```

## Output

The script generates several visualizations:
1. Bar chart of most common words
2. Word cloud visualization
3. Sentiment distribution histogram
4. Time series plot of tweet activity
5. Engagement metrics visualization

## Requirements

- Python 3.7+
- pandas
- nltk
- wordcloud
- textblob
- seaborn
- matplotlib

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch
3. Submit a pull request

## Author

[Your Name]

## Acknowledgments

- NLTK for natural language processing
- WordCloud for visualization
- TextBlob for sentiment analysis
