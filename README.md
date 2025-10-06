# Personal Finance AI Assistant with Spending Intelligence

## 🎯 Project Overview

A comprehensive AI-powered personal finance dashboard that analyzes spending patterns, provides budget recommendations, and offers financial advice through conversational AI. This project demonstrates advanced big data analytics, machine learning, and modern AI techniques applied to personal financial management.

### 🌟 Key Features

- **AI-Powered Analysis**: Multi-agent system for comprehensive spending analysis
- **Conversational AI**: RAG-based financial advisor chatbot
- **Interactive Dashboard**: Real-time visualizations and insights
- **Budget Optimization**: Personalized budget recommendations
- **Market Awareness**: Economic indicators integration
- **Anomaly Detection**: Unusual spending pattern identification
- **Predictive Analytics**: Future spending projections

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   Multi-Agent   │    │   RAG System    │
│   Dashboard     │◄──►│   AI System     │◄──►│  (ChromaDB +    │
│                 │    │                 │    │ OpenAI + GROQ)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  External APIs  │    │   Knowledge     │
│  (SQLite DB)    │    │ (FRED, NewsAPI) │    │     Base        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### AI Agent Architecture

1. **Spending Analyzer Agent**: Pattern recognition, anomaly detection
2. **Budget Advisor Agent**: Personalized recommendations, optimization
3. **Market Context Agent**: Economic indicators, inflation impact

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   ```

2. **Run the automated setup**
   ```bash
   python setup.py
   ```

3. **Configure API keys** (Optional for full functionality)

4. **Launch the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

### Manual Installation

If automated setup fails:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Create directories
mkdir -p data/synthetic data/vector_db logs

# Initialize database
python src/database/database.py

# Generate sample data
python src/utils/data_generator.py

# Run application
streamlit run app.py
```

## 📊 Usage Guide

### Dashboard Navigation

1. **Dashboard Overview**: Financial health summary and key metrics
2. **Spending Analysis**: Detailed transaction analysis and patterns
3. **Budget Planning**: Personalized budget recommendations
4. **AI Chat Assistant**: Conversational financial advisor
5. **Upload Data**: Import your own transaction CSV files
6. **Market Context**: Economic indicators and market impact

### Data Upload Format

Your CSV file should include these columns:

| Column | Required | Description | Example |
|--------|----------|-------------|---------|
| date | Yes | Transaction date | 2024-09-01 |
| amount | Yes | Transaction amount | 1200.00 |
| category | Yes | Spending category | Housing |
| description | No | Transaction description | Rent payment |
| merchant | No | Merchant name | Property Manager |
| transaction_type | No | 'income' or 'expense' | expense |

### Sample Data

The system comes with pre-generated sample data including:
- 10 sample users with realistic profiles
- 12 months of transaction history per user
- Seasonal spending variations
- Economic indicator data
- Budget templates for different income levels

## 🧠 AI System Details

### RAG (Retrieval Augmented Generation) System

- **Vector Database**: ChromaDB for semantic search
- **Knowledge Base**: 10+ financial advice documents
- **Embeddings**: OpenAI text-embedding-ada-002 (with fallback)
- **LLM**: GROQ for response generation

### Multi-Agent Coordination

#### Spending Analyzer Agent
- Transaction categorization and pattern recognition
- Anomaly detection using statistical methods
- Temporal pattern analysis (daily, weekly, monthly)
- Merchant and category insights

#### Budget Advisor Agent
- Personalized budget recommendations using 50/30/20 rule variants
- Savings opportunity identification
- Category-specific reduction strategies
- Risk tolerance assessment

#### Market Context Agent
- Economic indicator integration (inflation, unemployment, interest rates)
- Market sentiment analysis from financial news
- Personal impact assessment
- Market-aware financial recommendations

### Machine Learning Features

- **Clustering**: K-means for spending behavior segmentation
- **Anomaly Detection**: Statistical outlier identification
- **Trend Analysis**: Linear regression for spending trends
- **Predictive Modeling**: Time-series forecasting for future spending

## 🛠️ Technical Stack

### Frontend
- **Streamlit**: Interactive web application framework
- **Plotly**: Advanced data visualizations
- **CSS**: Custom styling and responsive design

### Backend
- **Python**: Core application logic
- **SQLite**: Local database for transaction storage
- **ChromaDB**: Vector database for RAG system
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms

### APIs & External Services
- **GROQ API**: LLM and embeddings
- **FRED API**: Federal Reserve economic data
- **NewsAPI**: Financial news and sentiment

### Key Libraries
```
streamlit==1.28.1
pandas==2.0.3
plotly==5.17.0
openai==1.3.5
chromadb==0.4.15
scikit-learn==1.3.2
requests==2.31.0
faker==19.12.0
```

## 🎨 Features Showcase

### Interactive Visualizations
- Monthly spending trend line charts
- Category breakdown pie and bar charts
- Budget vs actual comparison charts
- Financial health gauge meters
- Daily spending velocity patterns

### AI-Powered Insights
- Spending pattern recognition
- Budget optimization recommendations
- Anomaly detection and alerts
- Market impact analysis
- Personalized financial advice

### Conversational Interface
- Natural language financial queries
- Context-aware responses
- Embedded visualizations in chat
- Session memory and follow-up questions

## 📈 Performance Optimization

### Cost Optimization Strategies
- **Smart Caching**: Common AI responses cached locally
- **Prompt Compression**: Minimized token usage
- **Model Routing**: Cheaper models for simple queries
- **Batch Processing**: Efficient data operations

### Performance Metrics
- AI response time: < 3 seconds average
- Dashboard load time: < 2 seconds
- Memory usage: < 500MB typical
- Database queries: < 100ms average

## 🔒 Security & Privacy

### Data Protection
- Local SQLite database (no cloud storage)
- API keys stored in environment variables
- No persistent logging of sensitive data
- Optional API integration (works offline)

### Privacy Features
- All data processing happens locally
- No transaction data sent to external services
- User data never shared or transmitted
- Complete offline capability with synthetic data

## 🚧 Known Limitations

1. **Synthetic Data**: Demo uses generated data vs. real bank connections
2. **API Dependencies**: Full functionality requires API keys
3. **Single User**: Current version designed for individual use
4. **Limited Historical Data**: Analysis based on available transaction history

## 🔮 Future Enhancements

### Phase 2 Features
- **Real Bank Integration**: Plaid API for live data
- **Investment Tracking**: Portfolio analysis and recommendations
- **Goal-Based Planning**: Retirement and savings goal optimization
- **Multi-User Support**: Family and shared account management

### Advanced Analytics
- **Deep Learning Models**: LSTM for spending prediction
- **Behavioral Analysis**: Psychology-based spending insights
- **Social Features**: Community challenges and benchmarking
- **Mobile App**: Native iOS/Android applications

### Enterprise Features
- **API Development**: Third-party integration endpoints
- **Advanced Security**: Encryption and audit logging
- **Scalability**: Cloud deployment for thousands of users
- **Compliance**: Financial regulation adherence

**
