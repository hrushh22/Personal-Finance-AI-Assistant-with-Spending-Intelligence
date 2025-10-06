#!/usr/bin/env python3
"""
Personal Finance AI Assistant Setup Script
Initializes the entire project environment, databases, and sample data
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('setup.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def create_directories():
    """Create all necessary project directories"""
    logger = logging.getLogger(__name__)
    
    directories = [
        'data/raw',
        'data/processed', 
        'data/synthetic',
        'data/vector_db',
        'src/agents',
        'src/database',
        'src/utils',
        'src/visualization',
        'logs',
        'tests',
        'config',
        'assets/images',
        'notebooks'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def install_requirements():
    """Install Python requirements"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Installing Python requirements...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        logger.info("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing requirements: {e}")
        return False
    except FileNotFoundError:
        logger.error("requirements.txt not found. Please ensure it exists in the project root.")
        return False
    
    return True

def create_env_file():
    """Create .env file if it doesn't exist"""
    logger = logging.getLogger(__name__)
    
    env_file = Path('.env')
    if not env_file.exists():
        logger.info("Creating .env file...")
        
        env_content = """# API Keys (replace with your actual keys)
OPENAI_API_KEY=your_openai_api_key_here
FRED_API_KEY=your_fred_api_key_here
NEWS_API_KEY=your_news_api_key_here

# Database Configuration
DATABASE_PATH=data/finance_ai.db
VECTOR_DB_PATH=data/vector_db

# App Configuration
DEBUG=True
LOG_LEVEL=INFO
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
"""
        
        with open('.env', 'w') as f:
            f.write(env_content)
        
        logger.info("Created .env file. Please update it with your actual API keys.")
    else:
        logger.info(".env file already exists.")

def initialize_database():
    """Initialize the database with sample data"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Initializing database...")
        
        # Add src to path
        sys.path.append('src')
        
        from database.database import FinanceDB
        from utils.data_generator import FinancialDataGenerator
        
        # Initialize database
        db = FinanceDB()
        logger.info("Database initialized successfully!")
        
        # Generate sample data
        logger.info("Generating sample data...")
        generator = FinancialDataGenerator()
        users_df, transactions_df = generator.save_sample_data()
        
        logger.info(f"Generated data for {len(users_df)} users with {len(transactions_df)} transactions")
        
        # Load sample data into database
        for _, user in users_df.iterrows():
            user_id = db.add_user(
                username=user['username'],
                email=user['email'],
                monthly_income=user['monthly_income']
            )
            
            if user_id:
                user_transactions = transactions_df[transactions_df['user_id'] == user['user_id']]
                db.bulk_add_transactions(user_transactions, user_id)
                logger.info(f"Loaded {len(user_transactions)} transactions for user {user['username']}")
        
        logger.info("Database initialization completed!")
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False
    
    return True

def initialize_rag_system():
    """Initialize the RAG system with knowledge base"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Initializing RAG system...")
        
        sys.path.append('src')
        from utils.rag_system import FinancialRAGSystem
        
        # Initialize RAG system
        rag = FinancialRAGSystem()
        num_docs = rag.initialize_knowledge_base()
        
        logger.info(f"RAG system initialized with {num_docs} knowledge documents!")
        
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        return False
    
    return True

def create_sample_config():
    """Create sample configuration files"""
    logger = logging.getLogger(__name__)
    
    # Create streamlit config
    streamlit_config_dir = Path('.streamlit')
    streamlit_config_dir.mkdir(exist_ok=True)
    
    config_content = """[server]
port = 8501
address = "localhost"
maxUploadSize = 50

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[browser]
gatherUsageStats = false
"""
    
    config_file = streamlit_config_dir / 'config.toml'
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    logger.info("Created Streamlit configuration file")
    
    # Create logging config
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            },
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.StreamHandler",
            },
            "file": {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.FileHandler",
                "filename": "logs/app.log",
            },
        },
        "loggers": {
            "": {
                "handlers": ["default", "file"],
                "level": "INFO",
                "propagate": False
            }
        }
    }
    
    import json
    with open('config/logging.json', 'w') as f:
        json.dump(logging_config, f, indent=2)
    
    logger.info("Created logging configuration file")

def run_tests():
    """Run basic system tests"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Running basic system tests...")
        
        sys.path.append('src')
        
        # Test database connection
        from database.database import FinanceDB
        db = FinanceDB()
        test_user_id = db.add_user("test_user", "test@example.com", 5000)
        if test_user_id:
            logger.info("✓ Database test passed")
        else:
            logger.warning("⚠ Database test failed - user may already exist")
        
        # Test RAG system
        from utils.rag_system import FinancialRAGSystem
        rag = FinancialRAGSystem()
        test_response = rag.generate_response("What is budgeting?")
        if test_response:
            logger.info("✓ RAG system test passed")
        else:
            logger.error("✗ RAG system test failed")
        
        # Test API client
        from utils.api_client import APIClient
        api_client = APIClient()
        market_context = api_client.get_comprehensive_market_context()
        if market_context:
            logger.info("✓ API client test passed")
        else:
            logger.error("✗ API client test failed")
        
        # Test agents
        from agents.financial_agents import FinancialAgentCoordinator
        coordinator = FinancialAgentCoordinator()
        logger.info("✓ Agent system test passed")
        
        logger.info("All basic tests completed!")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        return False
    
    return True

def print_next_steps():
    """Print next steps for the user"""
    logger = logging.getLogger(__name__)
    
    next_steps = """
🎉 Personal Finance AI Assistant Setup Complete!

Next Steps:
1. Update your API keys in the .env file:
   - Get OpenAI API key from: https://platform.openai.com/api-keys
   - Get FRED API key from: https://fred.stlouisfed.org/docs/api/api_key.html
   - Get News API key from: https://newsapi.org/register

2. Run the application:
   streamlit run app.py

3. Open your browser and go to: http://localhost:8501

4. Upload your own transaction data or use the sample data to explore!

📁 Project Structure:
- app.py - Main Streamlit application
- src/ - Source code modules
- data/ - Database and data files
- config/ - Configuration files
- logs/ - Application logs

🔧 Configuration:
- Database: data/finance_ai.db
- Vector DB: data/vector_db/
- Logs: logs/app.log
- Config: .streamlit/config.toml

🆘 Troubleshooting:
- Check logs/setup.log for setup issues
- Ensure all requirements are installed
- Update API keys in .env file for full functionality

Happy budgeting! 💰
"""
    
    print(next_steps)
    logger.info("Setup completed successfully!")

def main():
    """Main setup function"""
    logger = setup_logging()
    logger.info("Starting Personal Finance AI Assistant setup...")
    
    steps = [
        ("Creating directories", create_directories),
        ("Installing requirements", install_requirements),
        ("Creating environment file", create_env_file),
        ("Creating configuration files", create_sample_config),
        ("Initializing database", initialize_database),
        ("Initializing RAG system", initialize_rag_system),
        ("Running tests", run_tests)
    ]
    
    for step_name, step_function in steps:
        logger.info(f"Step: {step_name}")
        try:
            if not step_function():
                logger.error(f"Failed at step: {step_name}")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Error in step '{step_name}': {e}")
            sys.exit(1)
    
    print_next_steps()

if __name__ == "__main__":
    main()