import chromadb
from groq import Groq
import pandas as pd
import os
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any
import json
import hashlib
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

class FinancialRAGSystem:
    def __init__(self, persist_directory="data/vector_db"):
        """Initialize the RAG system with ChromaDB and Groq"""
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize Groq
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            logging.warning("GROQ_API_KEY not found. Using fallback responses.")
            self.groq_client = None
        else:
            self.groq_client = Groq(api_key=self.groq_api_key)
        
        # Initialize FREE local embeddings
        logging.info("Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection("financial_knowledge")
        except:
            self.collection = self.client.create_collection(
                name="financial_knowledge",
                metadata={"description": "Financial advice and knowledge base"}
            )
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding using FREE local model"""
        return self.embedding_model.encode(text).tolist()
    
    def add_financial_knowledge(self, documents: List[Dict[str, Any]]):
        """Add financial knowledge documents to the vector database"""
        texts = []
        metadatas = []
        ids = []
        embeddings = []
        
        for i, doc in enumerate(documents):
            doc_id = f"doc_{i}_{hashlib.md5(doc['content'].encode()).hexdigest()[:8]}"
            
            texts.append(doc['content'])
            metadatas.append({
                'title': doc.get('title', ''),
                'category': doc.get('category', 'general'),
                'source': doc.get('source', 'internal'),
                'created_at': datetime.now().isoformat()
            })
            ids.append(doc_id)
            embeddings.append(self.get_embedding(doc['content']))
        
        # Add to collection
        try:
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            self.logger.info(f"Added {len(documents)} documents to knowledge base")
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
    
    def search_knowledge(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search for relevant knowledge given a query"""
        try:
            query_embedding = self.get_embedding(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            relevant_docs = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    relevant_docs.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0
                    })
            
            return relevant_docs
            
        except Exception as e:
            self.logger.error(f"Error searching knowledge base: {e}")
            return []
    
    def generate_response(self, query: str, user_context: Dict = None) -> str:
        """Generate response using RAG approach with Groq"""
        # Search for relevant knowledge
        relevant_docs = self.search_knowledge(query, n_results=3)
        
        # Build context from retrieved documents
        context_parts = []
        for doc in relevant_docs:
            context_parts.append(f"Knowledge: {doc['content']}")
        
        context = "\n\n".join(context_parts) if context_parts else "No specific knowledge found."
        
        # Add user context if provided
        user_info = ""
        if user_context:
            user_info = f"""
User Context:
- Monthly Income: ${user_context.get('monthly_income', 'Not specified')}
- Recent Spending: {user_context.get('recent_spending_summary', 'No data')}
- Budget Status: {user_context.get('budget_status', 'No budget set')}
"""
        
        # Create prompt
        prompt = f"""You are a helpful financial advisor AI assistant. Use the provided knowledge and user context to answer the user's question.

{context}

{user_info}

User Question: {query}

Provide a helpful, accurate, and personalized response based on the knowledge above. If the knowledge doesn't directly address the question, provide general financial advice. Keep the response conversational and actionable.

Response:"""
        
        try:
            if self.groq_client:
                chat_completion = self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a knowledgeable financial advisor."},
                        {"role": "user", "content": prompt}
                    ],
                    model="llama-3.1-8b-instant",
                    max_tokens=500,
                    temperature=0.7
                )
                return chat_completion.choices[0].message.content.strip()
            else:
                # Fallback response
                return self._generate_fallback_response(query, relevant_docs, user_context)
                
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return self._generate_fallback_response(query, relevant_docs, user_context)
    
    def _generate_fallback_response(self, query: str, relevant_docs: List[Dict], user_context: Dict = None) -> str:
        """Generate a fallback response when Groq is not available"""
        query_lower = query.lower()
        
        # Simple keyword-based responses
        if any(word in query_lower for word in ['budget', 'budgeting']):
            return """Here are some budgeting tips:
1. Use the 50/30/20 rule: 50% needs, 30% wants, 20% savings
2. Track all your expenses for at least a month
3. Set realistic limits for each spending category
4. Review and adjust your budget monthly
5. Use budgeting apps or spreadsheets to stay organized"""
        
        elif any(word in query_lower for word in ['save', 'saving', 'savings']):
            return """Effective saving strategies:
1. Pay yourself first - save before spending
2. Automate your savings to make it effortless
3. Start with a small emergency fund ($500-$1000)
4. Aim to save 3-6 months of expenses eventually
5. Consider high-yield savings accounts for better returns"""
        
        elif any(word in query_lower for word in ['debt', 'credit card']):
            return """For managing debt:
1. List all debts with balances and interest rates
2. Pay minimum on all debts, extra on highest interest rate
3. Consider debt consolidation if it lowers your rate
4. Avoid taking on new debt while paying off existing debt
5. Create a debt payoff timeline and stick to it"""
        
        elif any(word in query_lower for word in ['invest', 'investing', 'investment']):
            return """Investment basics:
1. Start with emergency fund before investing
2. Contribute to employer 401(k) match if available
3. Consider low-cost index funds for beginners
4. Diversify across different asset classes
5. Stay consistent and don't try to time the market"""
        
        elif any(word in query_lower for word in ['spend', 'spending', 'expense']):
            response = "Here's advice on managing spending:\n"
            response += "1. Track where your money goes each month\n"
            response += "2. Identify your needs vs wants\n"
            response += "3. Use the 24-hour rule for non-essential purchases\n"
            
            if user_context and user_context.get('recent_spending_summary'):
                response += f"\nBased on your recent spending: {user_context['recent_spending_summary']}"
            
            return response
        
        else:
            # Generic financial advice
            return """I'm here to help with your financial questions! Here are some general principles:
1. Create and stick to a budget
2. Build an emergency fund
3. Pay off high-interest debt
4. Save for retirement early and consistently
5. Invest in your financial education

Feel free to ask me about specific topics like budgeting, saving, investing, or debt management."""
    
    def initialize_knowledge_base(self):
        """Initialize the knowledge base with financial wisdom"""
        financial_knowledge = [
            {
                'title': 'Emergency Fund Basics',
                'category': 'savings',
                'content': 'An emergency fund is money set aside for unexpected expenses like medical bills, car repairs, or job loss. Financial experts recommend saving 3-6 months of living expenses in a easily accessible account. Start with $500-$1000 as your initial goal, then gradually build up. Keep emergency funds in high-yield savings accounts for better returns while maintaining liquidity.',
                'source': 'financial_planning'
            },
            {
                'title': '50/30/20 Budget Rule',
                'category': 'budgeting',
                'content': 'The 50/30/20 rule is a simple budgeting method: 50% of after-tax income for needs (housing, food, utilities, minimum debt payments), 30% for wants (entertainment, dining out, hobbies), and 20% for savings and debt repayment. This rule provides a balanced approach to spending while ensuring you save for the future.',
                'source': 'budgeting_guide'
            },
            {
                'title': 'Debt Avalanche vs Snowball',
                'category': 'debt_management',
                'content': 'Two popular debt repayment strategies: Debt Avalanche pays off highest interest rate debt first (mathematically optimal), while Debt Snowball pays off smallest balance first (psychologically motivating). Choose avalanche to minimize interest paid over time, or snowball for motivation through quick wins.',
                'source': 'debt_strategies'
            },
            {
                'title': 'Investment Basics for Beginners',
                'category': 'investing',
                'content': 'Before investing: have emergency fund and pay off high-interest debt. Start with employer 401(k) match, then consider low-cost index funds. Diversification reduces risk - dont put all money in one stock or sector. Time in market beats timing the market. Dollar-cost averaging helps reduce impact of market volatility.',
                'source': 'investment_guide'
            },
            {
                'title': 'Tracking Your Spending',
                'category': 'spending_analysis',
                'content': 'Track every expense for at least one month to understand spending patterns. Use apps, spreadsheets, or pen and paper. Categorize expenses as needs vs wants. Look for spending leaks - small recurring expenses that add up. Regular tracking helps identify areas to cut back and ensures youre staying within budget.',
                'source': 'expense_tracking'
            },
            {
                'title': 'Credit Score Improvement',
                'category': 'credit_management',
                'content': 'Credit scores range from 300-850. Factors affecting score: payment history (35%), credit utilization (30%), length of credit history (15%), types of credit (10%), new credit inquiries (10%). Pay bills on time, keep credit utilization below 30% (ideally under 10%), maintain old accounts, and limit new credit applications.',
                'source': 'credit_education'
            },
            {
                'title': 'Retirement Planning Basics',
                'category': 'retirement',
                'content': 'Start saving for retirement as early as possible due to compound interest. Contribute enough to get full employer match in 401(k) - its free money. Consider Roth IRA for tax-free withdrawals in retirement. Rule of thumb: save 10-15% of income for retirement. Increase contributions with salary raises.',
                'source': 'retirement_planning'
            },
            {
                'title': 'High-Yield Savings Accounts',
                'category': 'savings_products',
                'content': 'High-yield savings accounts offer better interest rates than traditional savings accounts. Look for accounts with competitive APY, no monthly fees, and FDIC insurance. Online banks often offer higher rates than traditional banks. Use for emergency funds and short-term savings goals.',
                'source': 'banking_products'
            },
            {
                'title': 'Understanding Inflation Impact',
                'category': 'economic_concepts',
                'content': 'Inflation erodes purchasing power over time. If inflation is 3% annually, $100 today will have the buying power of $97 next year. This is why investing is important - keeping money in low-interest accounts means losing value to inflation. Adjust spending and savings goals based on inflation rates.',
                'source': 'economic_education'
            },
            {
                'title': 'Financial Goal Setting',
                'category': 'planning',
                'content': 'Set SMART financial goals: Specific, Measurable, Achievable, Relevant, Time-bound. Examples: Save $5,000 for emergency fund in 12 months, or Pay off $3,000 credit card debt in 18 months. Break large goals into smaller milestones. Review and adjust goals regularly. Celebrate achievements to stay motivated.',
                'source': 'goal_setting'
            }
        ]
        
        self.add_financial_knowledge(financial_knowledge)
        return len(financial_knowledge)
    
    def get_contextual_advice(self, user_data: Dict, query: str = None) -> Dict:
        """Generate contextual advice based on user's financial data"""
        advice = {
            'insights': [],
            'recommendations': [],
            'warnings': []
        }
        
        # Analyze spending patterns
        if 'spending_by_category' in user_data:
            total_spending = user_data['spending_by_category']['total_spent'].sum()
            monthly_income = user_data.get('monthly_income', 0)
            
            if monthly_income > 0:
                spending_ratio = total_spending / monthly_income
                
                if spending_ratio > 0.9:
                    advice['warnings'].append(
                        "You're spending over 90% of your income. Consider reducing expenses or increasing income."
                    )
                elif spending_ratio > 0.8:
                    advice['insights'].append(
                        "You're spending 80%+ of income. Building an emergency fund may be challenging."
                    )
                
                # Category-specific advice
                for _, row in user_data['spending_by_category'].iterrows():
                    category = row['category']
                    amount = row['total_spent']
                    percentage = (amount / monthly_income) * 100
                    
                    if category == 'Food & Dining' and percentage > 20:
                        advice['recommendations'].append(
                            f"Food spending is {percentage:.1f}% of income. Consider meal planning to reduce costs."
                        )
                    elif category == 'Entertainment' and percentage > 10:
                        advice['recommendations'].append(
                            f"Entertainment spending is {percentage:.1f}% of income. Look for free activities to reduce costs."
                        )
                    elif category == 'Housing' and percentage > 35:
                        advice['warnings'].append(
                            f"Housing costs are {percentage:.1f}% of income. Recommended maximum is 30%."
                        )
        
        # Budget analysis
        if 'budget_status' in user_data:
            for _, budget in user_data['budget_status'].iterrows():
                if budget['percentage_used'] > 100:
                    advice['warnings'].append(
                        f"You've exceeded your {budget['category']} budget by {budget['percentage_used']-100:.1f}%"
                    )
                elif budget['percentage_used'] > 80:
                    advice['insights'].append(
                        f"You're at {budget['percentage_used']:.1f}% of your {budget['category']} budget"
                    )
        
        return advice
    
    def chat_with_context(self, query: str, user_data: Dict) -> Dict:
        """Enhanced chat function with user context and advice"""
        # Get contextual advice
        contextual_advice = self.get_contextual_advice(user_data)
        
        # Prepare user context for RAG
        user_context = {
            'monthly_income': user_data.get('monthly_income'),
            'recent_spending_summary': self._summarize_spending(user_data),
            'budget_status': self._summarize_budget_status(user_data)
        }
        
        # Generate response
        response = self.generate_response(query, user_context)
        
        return {
            'response': response,
            'contextual_advice': contextual_advice,
            'relevant_insights': self._get_relevant_insights(query, user_data)
        }
    
    def _summarize_spending(self, user_data: Dict) -> str:
        """Summarize user's recent spending"""
        if 'spending_by_category' not in user_data:
            return "No spending data available"
        
        spending_df = user_data['spending_by_category']
        top_categories = spending_df.nlargest(3, 'total_spent')
        
        summary_parts = []
        for _, row in top_categories.iterrows():
            summary_parts.append(f"{row['category']}: ${row['total_spent']:.2f}")
        
        return "Top spending: " + ", ".join(summary_parts)
    
    def _summarize_budget_status(self, user_data: Dict) -> str:
        """Summarize budget status"""
        if 'budget_status' not in user_data:
            return "No budget set"
        
        budget_df = user_data['budget_status']
        over_budget = budget_df[budget_df['percentage_used'] > 100]
        
        if len(over_budget) > 0:
            return f"Over budget in {len(over_budget)} categories"
        else:
            avg_usage = budget_df['percentage_used'].mean()
            return f"Average budget usage: {avg_usage:.1f}%"
    
    def _get_relevant_insights(self, query: str, user_data: Dict) -> List[str]:
        """Get insights relevant to the user's query and data"""
        insights = []
        query_lower = query.lower()
        
        if 'spending_by_category' in user_data:
            spending_df = user_data['spending_by_category']
            
            # Query about specific categories
            for _, row in spending_df.iterrows():
                category = row['category'].lower()
                if category in query_lower:
                    insights.append(f"Your {row['category']} spending: ${row['total_spent']:.2f} across {row['transaction_count']} transactions")
        
        return insights[:3]  # Limit to top 3 insights

# Usage example and testing
if __name__ == "__main__":
    # Initialize RAG system
    rag = FinancialRAGSystem()
    
    # Initialize knowledge base
    num_docs = rag.initialize_knowledge_base()
    print(f"Initialized knowledge base with {num_docs} documents")
    
    # Test search
    query = "How should I create a budget?"
    results = rag.search_knowledge(query)
    print(f"\nSearch results for '{query}':")
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc['metadata'].get('title', 'Untitled')}")
        print(f"   Content: {doc['content'][:100]}...")
    
    # Test response generation
    response = rag.generate_response(query)
    print(f"\nGenerated response:\n{response}")
    
    # Test with user context
    sample_user_data = {
        'monthly_income': 5000,
        'spending_by_category': pd.DataFrame([
            {'category': 'Food & Dining', 'total_spent': 800, 'transaction_count': 25},
            {'category': 'Housing', 'total_spent': 1500, 'transaction_count': 1},
            {'category': 'Transportation', 'total_spent': 400, 'transaction_count': 8}
        ])
    }
    
    chat_response = rag.chat_with_context("Why am I spending so much on food?", sample_user_data)
    print(f"\nContextual chat response:\n{chat_response['response']}")
    
    if chat_response['contextual_advice']['recommendations']:
        print(f"\nRecommendations:")
        for rec in chat_response['contextual_advice']['recommendations']:
            print(f"- {rec}")