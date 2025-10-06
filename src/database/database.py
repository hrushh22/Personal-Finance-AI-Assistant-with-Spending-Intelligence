import sqlite3
import pandas as pd
import os
from datetime import datetime
import logging

class FinanceDB:
    def __init__(self, db_path="data/finance_ai.db"):
        """Initialize database connection and create tables if they don't exist"""
        self.db_path = db_path
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self.init_database()
        
    def init_database(self):
        """Create all necessary tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                monthly_income REAL,
                financial_goals TEXT
            )
        ''')
        
        # Transactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                date DATE NOT NULL,
                amount REAL NOT NULL,
                category TEXT NOT NULL,
                subcategory TEXT,
                description TEXT,
                merchant TEXT,
                transaction_type TEXT CHECK(transaction_type IN ('income', 'expense')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Categories table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS categories (
                category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_name TEXT UNIQUE NOT NULL,
                parent_category TEXT,
                budget_percentage REAL,
                is_essential BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Budgets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS budgets (
                budget_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                category TEXT NOT NULL,
                monthly_limit REAL NOT NULL,
                current_spent REAL DEFAULT 0,
                month_year TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # AI Insights table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_insights (
                insight_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                insight_type TEXT NOT NULL,
                insight_text TEXT NOT NULL,
                confidence_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Economic Data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS economic_data (
                data_id INTEGER PRIMARY KEY AUTOINCREMENT,
                indicator_name TEXT NOT NULL,
                value REAL NOT NULL,
                date DATE NOT NULL,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Chat History table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                user_message TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_user_date ON transactions(user_id, date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_budgets_user_month ON budgets(user_id, month_year)')
        
        # Insert default categories
        default_categories = [
            ('Housing', 'Essential', 30.0, True),
            ('Food & Dining', 'Essential', 15.0, True),
            ('Transportation', 'Essential', 15.0, True),
            ('Utilities', 'Essential', 10.0, True),
            ('Healthcare', 'Essential', 5.0, True),
            ('Insurance', 'Essential', 5.0, True),
            ('Entertainment', 'Discretionary', 5.0, False),
            ('Shopping', 'Discretionary', 5.0, False),
            ('Travel', 'Discretionary', 5.0, False),
            ('Savings', 'Financial', 5.0, True),
            ('Investments', 'Financial', 0.0, False)
        ]
        
        cursor.executemany('''
            INSERT OR IGNORE INTO categories (category_name, parent_category, budget_percentage, is_essential)
            VALUES (?, ?, ?, ?)
        ''', default_categories)
        
        conn.commit()
        conn.close()
        
        logging.info("Database initialized successfully")
    
    def add_user(self, username, email=None, monthly_income=None):
        """Add a new user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (username, email, monthly_income)
                VALUES (?, ?, ?)
            ''', (username, email, monthly_income))
            
            user_id = cursor.lastrowid
            conn.commit()
            return user_id
        except sqlite3.IntegrityError:
            return None
        finally:
            conn.close()
    
    def add_transaction(self, user_id, date, amount, category, description="", merchant="", transaction_type="expense"):
        """Add a new transaction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO transactions (user_id, date, amount, category, description, merchant, transaction_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, date, amount, category, description, merchant, transaction_type))
        
        conn.commit()
        conn.close()
    
    def bulk_add_transactions(self, transactions_df, user_id):
        """Add multiple transactions from DataFrame"""
        conn = sqlite3.connect(self.db_path)
        
        # Add user_id to DataFrame
        transactions_df['user_id'] = user_id
        
        # Ensure required columns exist
        required_columns = ['user_id', 'date', 'amount', 'category', 'description', 'merchant', 'transaction_type']
        for col in required_columns:
            if col not in transactions_df.columns:
                transactions_df[col] = ''
        
        # Insert data
        transactions_df[required_columns].to_sql('transactions', conn, if_exists='append', index=False)
        
        conn.close()
        logging.info(f"Added {len(transactions_df)} transactions for user {user_id}")
    
    def get_user_transactions(self, user_id, start_date=None, end_date=None):
        """Get transactions for a user within date range"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM transactions WHERE user_id = ?"
        params = [user_id]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_spending_by_category(self, user_id, start_date=None, end_date=None):
        """Get spending summary by category"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT category, 
                   SUM(amount) as total_spent,
                   COUNT(*) as transaction_count,
                   AVG(amount) as avg_transaction
            FROM transactions 
            WHERE user_id = ? AND transaction_type = 'expense'
        '''
        params = [user_id]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " GROUP BY category ORDER BY total_spent DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def add_budget(self, user_id, category, monthly_limit, month_year):
        """Add or update budget for a category"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if budget exists
        cursor.execute('''
            SELECT budget_id FROM budgets 
            WHERE user_id = ? AND category = ? AND month_year = ?
        ''', (user_id, category, month_year))
        
        if cursor.fetchone():
            # Update existing budget
            cursor.execute('''
                UPDATE budgets SET monthly_limit = ?
                WHERE user_id = ? AND category = ? AND month_year = ?
            ''', (monthly_limit, user_id, category, month_year))
        else:
            # Create new budget
            cursor.execute('''
                INSERT INTO budgets (user_id, category, monthly_limit, month_year)
                VALUES (?, ?, ?, ?)
            ''', (user_id, category, monthly_limit, month_year))
        
        conn.commit()
        conn.close()
    
    def get_budget_status(self, user_id, month_year=None):
        """Get budget vs actual spending"""
        if not month_year:
            month_year = datetime.now().strftime("%Y-%m")
        
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT b.category, b.monthly_limit,
                   COALESCE(SUM(t.amount), 0) as actual_spent,
                   (COALESCE(SUM(t.amount), 0) / b.monthly_limit * 100) as percentage_used
            FROM budgets b
            LEFT JOIN transactions t ON b.category = t.category 
                AND t.user_id = b.user_id 
                AND strftime('%Y-%m', t.date) = b.month_year
                AND t.transaction_type = 'expense'
            WHERE b.user_id = ? AND b.month_year = ?
            GROUP BY b.category, b.monthly_limit
        '''
        
        df = pd.read_sql_query(query, conn, params=[user_id, month_year])
        conn.close()
        
        return df
    
    def add_ai_insight(self, user_id, insight_type, insight_text, confidence_score=None):
        """Add AI-generated insight"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO ai_insights (user_id, insight_type, insight_text, confidence_score)
            VALUES (?, ?, ?, ?)
        ''', (user_id, insight_type, insight_text, confidence_score))
        
        conn.commit()
        conn.close()
    
    def save_chat_message(self, user_id, user_message, ai_response, session_id=None):
        """Save chat conversation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO chat_history (user_id, user_message, ai_response, session_id)
            VALUES (?, ?, ?, ?)
        ''', (user_id, user_message, ai_response, session_id))
        
        conn.commit()
        conn.close()
    
    def get_chat_history(self, user_id, limit=10):
        """Get recent chat history"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT user_message, ai_response, timestamp
            FROM chat_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=[user_id, limit])
        conn.close()
        
        return df.iloc[::-1]  # Reverse to get chronological order

# Usage example and testing
if __name__ == "__main__":
    # Initialize database
    db = FinanceDB()
    
    # Add a test user
    user_id = db.add_user("john_doe", "john@example.com", 5000.0)
    if user_id:
        print(f"Created user with ID: {user_id}")
        
        # Add some test transactions
        test_transactions = [
            (user_id, "2024-09-01", 1200.0, "Housing", "Rent payment", "Property Manager", "expense"),
            (user_id, "2024-09-02", 45.50, "Food & Dining", "Grocery shopping", "Whole Foods", "expense"),
            (user_id, "2024-09-03", 25.00, "Transportation", "Gas", "Shell", "expense"),
            (user_id, "2024-09-04", 5000.0, "Salary", "Monthly salary", "Company", "income")
        ]
        
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        cursor.executemany('''
            INSERT INTO transactions (user_id, date, amount, category, description, merchant, transaction_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', test_transactions)
        conn.commit()
        conn.close()
        
        print("Added test transactions")
        
        # Test queries
        transactions = db.get_user_transactions(user_id)
        print(f"User has {len(transactions)} transactions")
        
        spending = db.get_spending_by_category(user_id)
        print("Spending by category:")
        print(spending)
    else:
        print("User already exists or error occurred")