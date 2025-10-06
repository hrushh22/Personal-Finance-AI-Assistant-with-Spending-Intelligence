import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging
import time
import json

# Load environment variables
load_dotenv()

class APIClient:
    def __init__(self):
        """Initialize API client with credentials"""
        self.fred_api_key = os.getenv('FRED_API_KEY')
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.openai_api_key = os.getenv('GROQ_API_KEY')
        
        # API endpoints
        self.fred_base_url = "https://api.stlouisfed.org/fred"
        self.news_base_url = "https://newsapi.org/v2"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Cache for API responses
        self.cache = {}
        self.cache_duration = 3600  # 1 hour in seconds
    
    def _is_cache_valid(self, key):
        """Check if cached data is still valid"""
        if key not in self.cache:
            return False
        
        cached_time = self.cache[key]['timestamp']
        return (time.time() - cached_time) < self.cache_duration
    
    def _cache_response(self, key, data):
        """Cache API response"""
        self.cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
    
    def get_economic_indicators(self, indicators=None, start_date=None, end_date=None):
        """Fetch economic indicators from FRED API"""
        if not self.fred_api_key:
            self.logger.warning("FRED API key not found. Using synthetic data.")
            return self._get_synthetic_economic_data(start_date, end_date)
        
        if indicators is None:
            indicators = {
                'CPIAUCSL': 'CPI_All_Items',  # Consumer Price Index
                'UNRATE': 'Unemployment_Rate',  # Unemployment Rate
                'FEDFUNDS': 'Federal_Funds_Rate',  # Federal Funds Rate
                'UMCSENT': 'Consumer_Sentiment'  # Consumer Sentiment
            }
        
        all_data = []
        
        for fred_code, indicator_name in indicators.items():
            cache_key = f"fred_{fred_code}_{start_date}_{end_date}"
            
            if self._is_cache_valid(cache_key):
                data = self.cache[cache_key]['data']
                self.logger.info(f"Using cached data for {indicator_name}")
            else:
                try:
                    # Build API URL
                    url = f"{self.fred_base_url}/series/observations"
                    params = {
                        'series_id': fred_code,
                        'api_key': self.fred_api_key,
                        'file_type': 'json',
                        'observation_start': start_date or '2024-01-01',
                        'observation_end': end_date or datetime.now().strftime('%Y-%m-%d')
                    }
                    
                    response = requests.get(url, params=params, timeout=30)
                    response.raise_for_status()
                    
                    data = response.json()
                    self._cache_response(cache_key, data)
                    self.logger.info(f"Fetched {indicator_name} data from FRED API")
                    
                    # Rate limiting
                    time.sleep(0.5)
                    
                except requests.exceptions.RequestException as e:
                    self.logger.error(f"Error fetching {indicator_name}: {e}")
                    continue
            
            # Process observations
            if 'observations' in data:
                for obs in data['observations']:
                    if obs['value'] != '.':  # FRED uses '.' for missing values
                        all_data.append({
                            'date': obs['date'],
                            'indicator_name': indicator_name,
                            'value': float(obs['value']),
                            'source': 'FRED'
                        })
        
        return pd.DataFrame(all_data)
    
    def get_financial_news(self, query='personal finance', language='en', page_size=20):
        """Fetch financial news from News API"""
        if not self.news_api_key:
            self.logger.warning("News API key not found. Using synthetic news data.")
            return self._get_synthetic_news_data()
        
        cache_key = f"news_{query}_{datetime.now().strftime('%Y-%m-%d')}"
        
        if self._is_cache_valid(cache_key):
            articles = self.cache[cache_key]['data']
            self.logger.info("Using cached news data")
        else:
            try:
                url = f"{self.news_base_url}/everything"
                params = {
                    'q': query,
                    'apiKey': self.news_api_key,
                    'language': language,
                    'sortBy': 'publishedAt',
                    'pageSize': page_size,
                    'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
                }
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                articles = data.get('articles', [])
                
                self._cache_response(cache_key, articles)
                self.logger.info(f"Fetched {len(articles)} news articles")
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error fetching news: {e}")
                return self._get_synthetic_news_data()
        
        # Process articles
        processed_articles = []
        for article in articles:
            processed_articles.append({
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'content': article.get('content', '')[:500] + '...' if article.get('content') else '',
                'url': article.get('url', ''),
                'published_at': article.get('publishedAt', ''),
                'source': article.get('source', {}).get('name', ''),
                'sentiment_score': self._analyze_sentiment(article.get('title', '') + ' ' + article.get('description', ''))
            })
        
        return pd.DataFrame(processed_articles)
    
    def _analyze_sentiment(self, text):
        """Simple sentiment analysis using TextBlob"""
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            return blob.sentiment.polarity  # Returns value between -1 and 1
        except ImportError:
            # Fallback: simple keyword-based sentiment
            positive_words = ['good', 'great', 'excellent', 'positive', 'growth', 'increase', 'profit']
            negative_words = ['bad', 'poor', 'negative', 'decline', 'decrease', 'loss', 'crisis']
            
            text_lower = text.lower()
            positive_score = sum(1 for word in positive_words if word in text_lower)
            negative_score = sum(1 for word in negative_words if word in text_lower)
            
            if positive_score + negative_score == 0:
                return 0.0
            return (positive_score - negative_score) / (positive_score + negative_score)
    
    def get_market_sentiment(self):
        """Get overall market sentiment from financial news"""
        news_df = self.get_financial_news(query='stock market economy inflation')
        
        if news_df.empty:
            return {
                'overall_sentiment': 0.0,
                'sentiment_description': 'Neutral',
                'key_topics': ['Economic uncertainty', 'Market volatility'],
                'news_count': 0
            }
        
        avg_sentiment = news_df['sentiment_score'].mean()
        
        if avg_sentiment > 0.1:
            sentiment_desc = 'Positive'
        elif avg_sentiment < -0.1:
            sentiment_desc = 'Negative'
        else:
            sentiment_desc = 'Neutral'
        
        # Extract key topics (simplified)
        all_titles = ' '.join(news_df['title'].fillna('').tolist())
        key_words = ['inflation', 'interest rates', 'employment', 'GDP', 'stocks', 'economy']
        key_topics = [word for word in key_words if word in all_titles.lower()]
        
        return {
            'overall_sentiment': round(avg_sentiment, 3),
            'sentiment_description': sentiment_desc,
            'key_topics': key_topics[:5],  # Top 5 topics
            'news_count': len(news_df),
            'recent_headlines': news_df['title'].head(3).tolist()
        }
    
    def _get_synthetic_economic_data(self, start_date=None, end_date=None):
        """Generate synthetic economic data when API is not available"""
        import random
        
        if not start_date:
            start_date = '2024-01-01'
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
        
        data = []
        base_values = {
            'CPI_All_Items': 310.0,
            'Unemployment_Rate': 3.8,
            'Federal_Funds_Rate': 4.5,
            'Consumer_Sentiment': 85.0
        }
        
        for date in date_range:
            for indicator, base_value in base_values.items():
                # Add some random variation
                if indicator == 'CPI_All_Items':
                    value = base_value + random.uniform(-5, 10)
                elif indicator == 'Unemployment_Rate':
                    value = max(0, base_value + random.uniform(-0.5, 1.0))
                elif indicator == 'Federal_Funds_Rate':
                    value = max(0, base_value + random.uniform(-0.25, 0.5))
                else:  # Consumer_Sentiment
                    value = base_value + random.uniform(-15, 15)
                
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'indicator_name': indicator,
                    'value': round(value, 2),
                    'source': 'Synthetic'
                })
        
        self.logger.info("Generated synthetic economic data")
        return pd.DataFrame(data)
    
    def _get_synthetic_news_data(self):
        """Generate synthetic news data when API is not available"""
        synthetic_articles = [
            {
                'title': 'Federal Reserve Maintains Interest Rates Amid Economic Uncertainty',
                'description': 'The Fed keeps rates steady as inflation shows signs of cooling while employment remains strong.',
                'content': 'The Federal Reserve announced today that it will maintain current interest rates...',
                'url': 'https://example.com/fed-rates',
                'published_at': datetime.now().strftime('%Y-%m-%d'),
                'source': 'Financial Times',
                'sentiment_score': 0.1
            },
            {
                'title': 'Consumer Spending Rises Despite Inflation Concerns',
                'description': 'Retail sales show resilience as Americans continue spending on essential and discretionary items.',
                'content': 'Consumer spending patterns indicate a robust economy despite ongoing inflation concerns...',
                'url': 'https://example.com/consumer-spending',
                'published_at': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                'source': 'Reuters',
                'sentiment_score': 0.3
            },
            {
                'title': 'Housing Market Shows Mixed Signals Across Regions',
                'description': 'Home prices continue to rise in some areas while cooling in others, creating a complex market landscape.',
                'content': 'The housing market presents a tale of two markets with regional variations...',
                'url': 'https://example.com/housing-market',
                'published_at': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'),
                'source': 'Wall Street Journal',
                'sentiment_score': -0.1
            },
            {
                'title': 'Technology Sector Drives Job Growth in Q3',
                'description': 'Tech companies continue hiring despite broader economic headwinds, supporting employment numbers.',
                'content': 'The technology sector remains a bright spot in the employment landscape...',
                'url': 'https://example.com/tech-jobs',
                'published_at': (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d'),
                'source': 'Bloomberg',
                'sentiment_score': 0.4
            },
            {
                'title': 'Personal Finance Apps See Surge in Usage',
                'description': 'More Americans are turning to digital tools to manage their finances amid economic uncertainty.',
                'content': 'Financial technology applications are experiencing unprecedented growth...',
                'url': 'https://example.com/fintech-growth',
                'published_at': (datetime.now() - timedelta(days=4)).strftime('%Y-%m-%d'),
                'source': 'TechCrunch',
                'sentiment_score': 0.2
            }
        ]
        
        self.logger.info("Generated synthetic news data")
        return pd.DataFrame(synthetic_articles)
    
    def get_comprehensive_market_context(self):
        """Get comprehensive market context combining economic indicators and news"""
        # Get economic indicators
        economic_data = self.get_economic_indicators()
        
        # Get market sentiment from news
        sentiment_data = self.get_market_sentiment()
        
        # Process economic indicators
        if not economic_data.empty:
            latest_indicators = economic_data.groupby('indicator_name').last().reset_index()
            
            economic_summary = {}
            for _, row in latest_indicators.iterrows():
                economic_summary[row['indicator_name']] = {
                    'current_value': row['value'],
                    'date': row['date']
                }
        else:
            economic_summary = {}
        
        # Combine all context
        market_context = {
            'economic_indicators': economic_summary,
            'market_sentiment': sentiment_data,
            'context_summary': self._generate_context_summary(economic_summary, sentiment_data),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return market_context
    
    def _generate_context_summary(self, economic_data, sentiment_data):
        """Generate a summary of market context"""
        summary_parts = []
        
        # Economic indicators summary
        if 'Unemployment_Rate' in economic_data:
            unemployment = economic_data['Unemployment_Rate']['current_value']
            if unemployment < 4.0:
                summary_parts.append("Employment remains strong")
            elif unemployment > 6.0:
                summary_parts.append("Employment concerns persist")
            else:
                summary_parts.append("Employment is stable")
        
        if 'Federal_Funds_Rate' in economic_data:
            fed_rate = economic_data['Federal_Funds_Rate']['current_value']
            if fed_rate > 5.0:
                summary_parts.append("Interest rates are elevated")
            elif fed_rate < 2.0:
                summary_parts.append("Interest rates remain low")
            else:
                summary_parts.append("Interest rates are moderate")
        
        # Sentiment summary
        sentiment_desc = sentiment_data.get('sentiment_description', 'neutral')
        summary_parts.append(f"Market sentiment is {sentiment_desc.lower()}")
        
        # Key topics
        key_topics = sentiment_data.get('key_topics', [])
        if key_topics:
            summary_parts.append(f"Key concerns include {', '.join(key_topics[:2])}")
        
        return '. '.join(summary_parts) + '.'

# Usage example and testing
if __name__ == "__main__":
    api_client = APIClient()
    
    # Test economic indicators
    print("Fetching economic indicators...")
    economic_data = api_client.get_economic_indicators(
        start_date='2024-01-01',
        end_date='2024-09-30'
    )
    print(f"Retrieved {len(economic_data)} economic data points")
    print(economic_data.head())
    
    # Test financial news
    print("\nFetching financial news...")
    news_data = api_client.get_financial_news(query='personal finance budgeting')
    print(f"Retrieved {len(news_data)} news articles")
    print(news_data[['title', 'sentiment_score']].head())
    
    # Test market sentiment
    print("\nAnalyzing market sentiment...")
    sentiment = api_client.get_market_sentiment()
    print(f"Overall sentiment: {sentiment['sentiment_description']} ({sentiment['overall_sentiment']})")
    print(f"Key topics: {', '.join(sentiment['key_topics'])}")
    
    # Test comprehensive context
    print("\nGetting comprehensive market context...")
    context = api_client.get_comprehensive_market_context()
    print(f"Context summary: {context['context_summary']}")
    
    # Save sample data
    economic_data.to_csv('data/economic_indicators_live.csv', index=False)
    news_data.to_csv('data/financial_news.csv', index=False)
    
    with open('data/market_context.json', 'w') as f:
        json.dump(context, f, indent=2, default=str)
    
    print("\nData saved successfully!")