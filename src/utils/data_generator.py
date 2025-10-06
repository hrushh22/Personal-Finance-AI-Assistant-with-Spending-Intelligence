import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import json

fake = Faker()

class FinancialDataGenerator:
    def __init__(self):
        self.categories = {
            'Housing': {
                'subcategories': ['Rent', 'Mortgage', 'Property Tax', 'HOA Fees', 'Maintenance'],
                'merchants': ['Property Manager', 'Landlord', 'Bank of America', 'Wells Fargo', 'Home Depot'],
                'avg_amount': 1200,
                'frequency': 'monthly'
            },
            'Food & Dining': {
                'subcategories': ['Groceries', 'Restaurants', 'Fast Food', 'Coffee', 'Delivery'],
                'merchants': ['Whole Foods', 'Safeway', 'Starbucks', 'McDonalds', 'DoorDash', 'Uber Eats', 'Chipotle'],
                'avg_amount': 85,
                'frequency': 'weekly'
            },
            'Transportation': {
                'subcategories': ['Gas', 'Public Transit', 'Uber/Lyft', 'Car Payment', 'Parking'],
                'merchants': ['Shell', 'Chevron', 'Metro', 'Uber', 'Lyft', 'Honda Finance'],
                'avg_amount': 120,
                'frequency': 'weekly'
            },
            'Utilities': {
                'subcategories': ['Electric', 'Gas', 'Water', 'Internet', 'Phone'],
                'merchants': ['PG&E', 'Comcast', 'Verizon', 'AT&T', 'Water Company'],
                'avg_amount': 150,
                'frequency': 'monthly'
            },
            'Entertainment': {
                'subcategories': ['Movies', 'Streaming', 'Gaming', 'Sports', 'Music'],
                'merchants': ['Netflix', 'Spotify', 'AMC Theaters', 'Steam', 'PlayStation Store'],
                'avg_amount': 45,
                'frequency': 'weekly'
            },
            'Shopping': {
                'subcategories': ['Clothing', 'Electronics', 'Books', 'Home Goods', 'Personal Care'],
                'merchants': ['Amazon', 'Target', 'Best Buy', 'Macys', 'CVS', 'Walgreens'],
                'avg_amount': 75,
                'frequency': 'weekly'
            },
            'Healthcare': {
                'subcategories': ['Doctor Visit', 'Pharmacy', 'Dental', 'Insurance Premium'],
                'merchants': ['Kaiser Permanente', 'CVS Pharmacy', 'Dental Associates'],
                'avg_amount': 120,
                'frequency': 'monthly'
            },
            'Travel': {
                'subcategories': ['Flights', 'Hotels', 'Car Rental', 'Vacation'],
                'merchants': ['Southwest Airlines', 'Marriott', 'Hertz', 'Expedia'],
                'avg_amount': 300,
                'frequency': 'quarterly'
            }
        }
        
        self.income_sources = {
            'Salary': {'merchants': ['Tech Corp', 'Finance Inc', 'Healthcare LLC'], 'amount_range': (4000, 8000)},
            'Freelance': {'merchants': ['Client A', 'Client B', 'Upwork'], 'amount_range': (500, 2000)},
            'Investment': {'merchants': ['Dividends', 'Capital Gains', 'Interest'], 'amount_range': (100, 1000)}
        }
    
    def generate_user_profile(self):
        """Generate a realistic user profile"""
        profile = {
            'user_id': fake.random_int(min=1, max=10000),
            'username': fake.user_name(),
            'email': fake.email(),
            'age': fake.random_int(min=22, max=65),
            'monthly_income': fake.random_int(min=3000, max=12000),
            'financial_goals': fake.random_element(['Emergency Fund', 'House Down Payment', 'Retirement', 'Vacation']),
            'risk_tolerance': fake.random_element(['Conservative', 'Moderate', 'Aggressive']),
            'location': fake.city() + ', ' + fake.state()
        }
        return profile
    
    def generate_monthly_transactions(self, user_profile, year=2024, month=9):
        """Generate realistic transactions for a specific month"""
        transactions = []
        start_date = datetime(year, month, 1)
        
        # Calculate days in month
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)
        
        days_in_month = (end_date - start_date).days + 1
        
        # Generate income (typically 1-2 times per month)
        income_days = [1, 15] if days_in_month >= 15 else [1]
        for day in income_days:
            if day <= days_in_month:
                transaction_date = datetime(year, month, day)
                
                # Primary salary
                income_amount = user_profile['monthly_income'] / len(income_days)
                income_amount *= random.uniform(0.95, 1.05)  # Small variation
                
                transactions.append({
                    'date': transaction_date.strftime('%Y-%m-%d'),
                    'amount': round(income_amount, 2),
                    'category': 'Salary',
                    'subcategory': 'Salary',
                    'description': 'Monthly salary payment',
                    'merchant': random.choice(self.income_sources['Salary']['merchants']),
                    'transaction_type': 'income'
                })
                
                # Occasional freelance income
                if random.random() < 0.3:  # 30% chance
                    freelance_amount = random.uniform(*self.income_sources['Freelance']['amount_range'])
                    transactions.append({
                        'date': transaction_date.strftime('%Y-%m-%d'),
                        'amount': round(freelance_amount, 2),
                        'category': 'Freelance',
                        'subcategory': 'Freelance',
                        'description': 'Freelance project payment',
                        'merchant': random.choice(self.income_sources['Freelance']['merchants']),
                        'transaction_type': 'income'
                    })
        
        # Generate expenses based on categories
        for category, details in self.categories.items():
            frequency = details['frequency']
            avg_amount = details['avg_amount']
            
            # Adjust amount based on income level
            income_multiplier = user_profile['monthly_income'] / 6000  # Base income of 6k
            adjusted_amount = avg_amount * income_multiplier
            
            if frequency == 'monthly':
                # 1-3 transactions per month for monthly categories
                num_transactions = random.randint(1, 3)
                for _ in range(num_transactions):
                    day = random.randint(1, days_in_month)
                    transaction_date = datetime(year, month, day)
                    
                    amount = adjusted_amount * random.uniform(0.7, 1.4)
                    if category == 'Housing':  # Housing is more consistent
                        amount = adjusted_amount * random.uniform(0.95, 1.05)
                    
                    transactions.append(self._create_transaction(
                        transaction_date, amount, category, details
                    ))
            
            elif frequency == 'weekly':
                # 3-6 transactions per month for weekly categories
                num_transactions = random.randint(3, 6)
                for _ in range(num_transactions):
                    day = random.randint(1, days_in_month)
                    transaction_date = datetime(year, month, day)
                    
                    amount = (adjusted_amount / 4) * random.uniform(0.3, 2.0)
                    
                    transactions.append(self._create_transaction(
                        transaction_date, amount, category, details
                    ))
            
            elif frequency == 'quarterly':
                # Occasional large transactions
                if random.random() < 0.33:  # 33% chance per month
                    day = random.randint(1, days_in_month)
                    transaction_date = datetime(year, month, day)
                    
                    amount = adjusted_amount * random.uniform(0.5, 2.0)
                    
                    transactions.append(self._create_transaction(
                        transaction_date, amount, category, details
                    ))
        
        # Add some random/miscellaneous transactions
        num_misc = random.randint(5, 15)
        for _ in range(num_misc):
            day = random.randint(1, days_in_month)
            transaction_date = datetime(year, month, day)
            
            misc_categories = ['ATM Fee', 'Bank Fee', 'Subscription', 'Other']
            category = random.choice(misc_categories)
            
            amount = random.uniform(5, 50)
            
            transactions.append({
                'date': transaction_date.strftime('%Y-%m-%d'),
                'amount': round(amount, 2),
                'category': category,
                'subcategory': category,
                'description': fake.sentence(nb_words=4),
                'merchant': fake.company(),
                'transaction_type': 'expense'
            })
        
        return transactions
    
    def _create_transaction(self, date, amount, category, details):
        """Create a single transaction"""
        subcategory = random.choice(details['subcategories'])
        merchant = random.choice(details['merchants'])
        
        # Generate realistic description
        descriptions = {
            'Housing': [f'Rent payment', f'Mortgage payment', f'Property maintenance'],
            'Food & Dining': [f'Grocery shopping', f'Dinner', f'Lunch', f'Coffee'],
            'Transportation': [f'Gas fill-up', f'Car payment', f'Parking fee'],
            'Utilities': [f'Electric bill', f'Internet bill', f'Phone bill'],
            'Entertainment': [f'Movie tickets', f'Subscription', f'Concert'],
            'Shopping': [f'Online purchase', f'Retail shopping', f'Personal items'],
            'Healthcare': [f'Doctor visit', f'Prescription', f'Insurance'],
            'Travel': [f'Flight booking', f'Hotel stay', f'Vacation expense']
        }
        
        description = random.choice(descriptions.get(category, ['Payment', 'Purchase']))
        
        return {
            'date': date.strftime('%Y-%m-%d'),
            'amount': round(amount, 2),
            'category': category,
            'subcategory': subcategory,
            'description': description,
            'merchant': merchant,
            'transaction_type': 'expense'
        }
    
    def generate_year_data(self, user_profile, year=2024):
        """Generate a full year of transaction data"""
        all_transactions = []
        
        for month in range(1, 13):
            monthly_transactions = self.generate_monthly_transactions(user_profile, year, month)
            all_transactions.extend(monthly_transactions)
        
        return all_transactions
    
    def add_seasonal_variations(self, transactions):
        """Add seasonal spending patterns"""
        df = pd.DataFrame(transactions)
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        
        # Holiday season adjustments (November, December)
        holiday_months = [11, 12]
        for idx, row in df.iterrows():
            if row['month'] in holiday_months and row['category'] in ['Shopping', 'Food & Dining', 'Travel']:
                df.loc[idx, 'amount'] *= random.uniform(1.2, 1.8)
        
        # Summer travel increase
        summer_months = [6, 7, 8]
        for idx, row in df.iterrows():
            if row['month'] in summer_months and row['category'] == 'Travel':
                df.loc[idx, 'amount'] *= random.uniform(1.1, 1.5)
        
        # Back-to-school shopping
        if df['month'].isin([8, 9]).any():
            back_to_school = df[(df['month'].isin([8, 9])) & (df['category'] == 'Shopping')]
            df.loc[back_to_school.index, 'amount'] *= random.uniform(1.1, 1.3)
        
        return df.drop('month', axis=1).to_dict('records')
    
    def create_sample_datasets(self, num_users=5):
        """Create sample datasets for multiple users"""
        all_users = []
        all_transactions = []
        
        for i in range(num_users):
            # Generate user profile
            user_profile = self.generate_user_profile()
            user_profile['user_id'] = i + 1  # Sequential IDs
            all_users.append(user_profile)
            
            # Generate transactions for this user
            user_transactions = self.generate_year_data(user_profile, 2024)
            user_transactions = self.add_seasonal_variations(user_transactions)
            
            # Add user_id to each transaction
            for transaction in user_transactions:
                transaction['user_id'] = user_profile['user_id']
            
            all_transactions.extend(user_transactions)
        
        # Convert to DataFrames
        users_df = pd.DataFrame(all_users)
        transactions_df = pd.DataFrame(all_transactions)
        
        # Sort transactions by date
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        transactions_df = transactions_df.sort_values(['user_id', 'date'])
        
        return users_df, transactions_df
    
    def save_sample_data(self, output_dir='data/synthetic'):
        """Generate and save sample data files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate data
        users_df, transactions_df = self.create_sample_datasets(num_users=10)
        
        # Save to CSV files
        users_df.to_csv(f'{output_dir}/sample_users.csv', index=False)
        transactions_df.to_csv(f'{output_dir}/sample_transactions.csv', index=False)
        
        # Create individual user files for demo
        for user_id in users_df['user_id'].unique():
            user_transactions = transactions_df[transactions_df['user_id'] == user_id]
            user_info = users_df[users_df['user_id'] == user_id].iloc[0]
            
            # Save user-specific transaction file
            user_transactions.to_csv(f'{output_dir}/user_{user_id}_transactions.csv', index=False)
            
            # Create user profile JSON
            user_profile = {
                'user_id': int(user_info['user_id']),
                'username': user_info['username'],
                'email': user_info['email'],
                'monthly_income': float(user_info['monthly_income']),
                'financial_goals': user_info['financial_goals'],
                'total_transactions': len(user_transactions),
                'date_range': {
                    'start': user_transactions['date'].min().strftime('%Y-%m-%d'),
                    'end': user_transactions['date'].max().strftime('%Y-%m-%d')
                }
            }
            
            with open(f'{output_dir}/user_{user_id}_profile.json', 'w') as f:
                json.dump(user_profile, f, indent=2)
        
        print(f"Generated sample data for {len(users_df)} users")
        print(f"Total transactions: {len(transactions_df)}")
        print(f"Files saved to: {output_dir}/")
        
        return users_df, transactions_df
    
    def generate_economic_indicators(self, start_date='2024-01-01', end_date='2024-12-31'):
        """Generate synthetic economic indicator data"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
        
        indicators = []
        
        # Inflation rate (annual percentage)
        base_inflation = 3.2
        for date in date_range:
            inflation_rate = base_inflation + random.uniform(-0.5, 0.8)
            indicators.append({
                'date': date.strftime('%Y-%m-%d'),
                'indicator_name': 'Inflation_Rate',
                'value': round(inflation_rate, 2),
                'source': 'BLS'
            })
        
        # Unemployment rate
        base_unemployment = 3.8
        for date in date_range:
            unemployment_rate = base_unemployment + random.uniform(-0.3, 0.7)
            indicators.append({
                'date': date.strftime('%Y-%m-%d'),
                'indicator_name': 'Unemployment_Rate',
                'value': round(unemployment_rate, 2),
                'source': 'BLS'
            })
        
        # Interest rates
        base_interest = 4.5
        for date in date_range:
            interest_rate = base_interest + random.uniform(-0.25, 0.5)
            indicators.append({
                'date': date.strftime('%Y-%m-%d'),
                'indicator_name': 'Federal_Funds_Rate',
                'value': round(interest_rate, 2),
                'source': 'Federal_Reserve'
            })
        
        # Consumer confidence index
        base_confidence = 85
        for date in date_range:
            confidence = base_confidence + random.uniform(-10, 15)
            indicators.append({
                'date': date.strftime('%Y-%m-%d'),
                'indicator_name': 'Consumer_Confidence',
                'value': round(confidence, 1),
                'source': 'Conference_Board'
            })
        
        return pd.DataFrame(indicators)
    
    def create_budget_templates(self):
        """Create budget templates based on income levels"""
        income_levels = [
            {'min': 30000, 'max': 50000, 'level': 'Low'},
            {'min': 50000, 'max': 80000, 'level': 'Medium'},
            {'min': 80000, 'max': 120000, 'level': 'High'},
            {'min': 120000, 'max': 200000, 'level': 'Very High'}
        ]
        
        budget_templates = []
        
        for income_bracket in income_levels:
            # 50/30/20 rule adjustments based on income level
            if income_bracket['level'] == 'Low':
                allocations = {
                    'Housing': 35, 'Food & Dining': 15, 'Transportation': 15,
                    'Utilities': 8, 'Healthcare': 5, 'Insurance': 4,
                    'Entertainment': 3, 'Shopping': 5, 'Savings': 10
                }
            elif income_bracket['level'] == 'Medium':
                allocations = {
                    'Housing': 30, 'Food & Dining': 12, 'Transportation': 15,
                    'Utilities': 6, 'Healthcare': 4, 'Insurance': 3,
                    'Entertainment': 5, 'Shopping': 8, 'Savings': 15, 'Travel': 2
                }
            elif income_bracket['level'] == 'High':
                allocations = {
                    'Housing': 25, 'Food & Dining': 10, 'Transportation': 12,
                    'Utilities': 5, 'Healthcare': 3, 'Insurance': 3,
                    'Entertainment': 7, 'Shopping': 10, 'Savings': 20, 'Travel': 5
                }
            else:  # Very High
                allocations = {
                    'Housing': 20, 'Food & Dining': 8, 'Transportation': 10,
                    'Utilities': 4, 'Healthcare': 3, 'Insurance': 3,
                    'Entertainment': 10, 'Shopping': 12, 'Savings': 25, 'Travel': 5
                }
            
            for category, percentage in allocations.items():
                budget_templates.append({
                    'income_level': income_bracket['level'],
                    'income_min': income_bracket['min'],
                    'income_max': income_bracket['max'],
                    'category': category,
                    'recommended_percentage': percentage,
                    'is_essential': category in ['Housing', 'Food & Dining', 'Transportation', 'Utilities', 'Healthcare', 'Insurance']
                })
        
        return pd.DataFrame(budget_templates)

# Usage and testing
if __name__ == "__main__":
    generator = FinancialDataGenerator()
    
    # Generate sample data
    users_df, transactions_df = generator.save_sample_data()
    
    # Generate economic indicators
    economic_df = generator.generate_economic_indicators()
    economic_df.to_csv('data/synthetic/economic_indicators.csv', index=False)
    
    # Create budget templates
    budget_templates = generator.create_budget_templates()
    budget_templates.to_csv('data/synthetic/budget_templates.csv', index=False)
    
    print("\nSample user profile:")
    print(users_df.head(1).to_dict('records')[0])
    
    print("\nSample transactions:")
    print(transactions_df.head(5))
    
    print("\nTransaction summary by category:")
    category_summary = transactions_df[transactions_df['transaction_type'] == 'expense'].groupby('category')['amount'].agg(['count', 'sum', 'mean']).round(2)
    print(category_summary)
    
    print("\nEconomic indicators sample:")
    print(economic_df.head())
    
    print("\nBudget templates sample:")
    print(budget_templates[budget_templates['income_level'] == 'Medium'])