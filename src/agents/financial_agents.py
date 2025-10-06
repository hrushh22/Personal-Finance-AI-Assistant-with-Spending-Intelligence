import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class BaseAgent:
    """Base class for all financial agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"Agent.{name}")
        self.logger.setLevel(logging.INFO)
    
    def log_action(self, action: str, details: str = ""):
        """Log agent actions"""
        self.logger.info(f"{self.name}: {action} {details}")

class SpendingAnalyzerAgent(BaseAgent):
    """Agent responsible for analyzing spending patterns and detecting anomalies"""
    
    def __init__(self):
        super().__init__("SpendingAnalyzer")
        self.scaler = StandardScaler()
        
    def analyze_spending_patterns(self, transactions_df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive spending pattern analysis"""
        self.log_action("Starting spending pattern analysis")
        
        if transactions_df.empty:
            return {"error": "No transaction data available"}
        
        # Ensure date column is datetime
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        
        analysis = {
            'category_analysis': self._analyze_by_category(transactions_df),
            'temporal_patterns': self._analyze_temporal_patterns(transactions_df),
            'spending_velocity': self._calculate_spending_velocity(transactions_df),
            'anomalies': self._detect_anomalies(transactions_df),
            'merchant_analysis': self._analyze_merchants(transactions_df),
            'spending_trends': self._analyze_trends(transactions_df)
        }
        
        self.log_action("Completed spending pattern analysis")
        return analysis
    
    def _analyze_by_category(self, df: pd.DataFrame) -> Dict:
        """Analyze spending by category"""
        expense_df = df[df['transaction_type'] == 'expense'].copy()
        
        category_stats = expense_df.groupby('category').agg({
            'amount': ['sum', 'mean', 'count', 'std'],
            'date': ['min', 'max']
        }).round(2)
        
        # Flatten column names
        category_stats.columns = ['total_spent', 'avg_transaction', 'transaction_count', 'std_dev', 'first_transaction', 'last_transaction']
        category_stats = category_stats.reset_index()
        
        # Calculate percentage of total spending
        total_spending = category_stats['total_spent'].sum()
        category_stats['percentage_of_total'] = (category_stats['total_spent'] / total_spending * 100).round(2)
        
        return {
            'summary': category_stats.to_dict('records'),
            'top_categories': category_stats.nlargest(5, 'total_spent')[['category', 'total_spent', 'percentage_of_total']].to_dict('records'),
            'most_frequent': category_stats.nlargest(5, 'transaction_count')[['category', 'transaction_count']].to_dict('records')
        }
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze spending patterns over time"""
        expense_df = df[df['transaction_type'] == 'expense'].copy()
        
        # Daily spending
        daily_spending = expense_df.groupby(expense_df['date'].dt.date)['amount'].sum()
        
        # Weekly patterns
        expense_df['day_of_week'] = expense_df['date'].dt.day_name()
        weekly_pattern = expense_df.groupby('day_of_week')['amount'].sum().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        
        # Monthly patterns
        expense_df['month'] = expense_df['date'].dt.month_name()
        monthly_pattern = expense_df.groupby('month')['amount'].sum()
        
        # Hour of day (if time info available)
        hourly_pattern = None
        if expense_df['date'].dt.hour.sum() > 0:  # Check if time data exists
            hourly_pattern = expense_df.groupby(expense_df['date'].dt.hour)['amount'].sum()
        
        return {
            'daily_average': daily_spending.mean(),
            'daily_std': daily_spending.std(),
            'highest_spending_day': {
                'date': str(daily_spending.idxmax()),
                'amount': daily_spending.max()
            },
            'weekly_pattern': weekly_pattern.to_dict(),
            'monthly_pattern': monthly_pattern.to_dict(),
            'hourly_pattern': hourly_pattern.to_dict() if hourly_pattern is not None else None
        }
    
    def _calculate_spending_velocity(self, df: pd.DataFrame) -> Dict:
        """Calculate spending velocity metrics"""
        expense_df = df[df['transaction_type'] == 'expense'].copy()
        
        if len(expense_df) < 2:
            return {"error": "Insufficient data for velocity calculation"}
        
        # Sort by date
        expense_df = expense_df.sort_values('date')
        
        # Calculate days between transactions
        expense_df['days_since_last'] = expense_df['date'].diff().dt.days
        
        # Remove first row (no previous transaction)
        velocity_data = expense_df[1:].copy()
        
        # Calculate velocity (amount per day between transactions)
        velocity_data['velocity'] = velocity_data['amount'] / velocity_data['days_since_last'].replace(0, 1)
        
        return {
            'avg_velocity': velocity_data['velocity'].mean(),
            'max_velocity': velocity_data['velocity'].max(),
            'velocity_trend': self._calculate_trend(velocity_data['velocity'].values),
            'avg_days_between_transactions': velocity_data['days_since_last'].mean()
        }
    
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict:
        """Detect anomalous transactions"""
        expense_df = df[df['transaction_type'] == 'expense'].copy()
        
        if len(expense_df) < 10:
            return {"error": "Insufficient data for anomaly detection"}
        
        anomalies = []
        
        # Statistical anomalies (transactions > 2 standard deviations)
        mean_amount = expense_df['amount'].mean()
        std_amount = expense_df['amount'].std()
        threshold = mean_amount + (2 * std_amount)
        
        statistical_anomalies = expense_df[expense_df['amount'] > threshold]
        
        for _, transaction in statistical_anomalies.iterrows():
            anomalies.append({
                'type': 'statistical_outlier',
                'date': transaction['date'].strftime('%Y-%m-%d'),
                'amount': transaction['amount'],
                'category': transaction['category'],
                'description': transaction.get('description', ''),
                'deviation': (transaction['amount'] - mean_amount) / std_amount
            })
        
        # Category-specific anomalies
        for category in expense_df['category'].unique():
            cat_data = expense_df[expense_df['category'] == category]
            if len(cat_data) >= 5:
                cat_mean = cat_data['amount'].mean()
                cat_std = cat_data['amount'].std()
                cat_threshold = cat_mean + (2 * cat_std)
                
                cat_anomalies = cat_data[cat_data['amount'] > cat_threshold]
                
                for _, transaction in cat_anomalies.iterrows():
                    if transaction['amount'] <= threshold:  # Avoid duplicates
                        anomalies.append({
                            'type': 'category_outlier',
                            'date': transaction['date'].strftime('%Y-%m-%d'),
                            'amount': transaction['amount'],
                            'category': transaction['category'],
                            'description': transaction.get('description', ''),
                            'category_average': cat_mean
                        })
        
        return {
            'count': len(anomalies),
            'anomalies': sorted(anomalies, key=lambda x: x['amount'], reverse=True)[:10]  # Top 10
        }
    
    def _analyze_merchants(self, df: pd.DataFrame) -> Dict:
        """Analyze spending by merchant"""
        expense_df = df[df['transaction_type'] == 'expense'].copy()
        
        merchant_stats = expense_df.groupby('merchant').agg({
            'amount': ['sum', 'count', 'mean']
        }).round(2)
        
        merchant_stats.columns = ['total_spent', 'transaction_count', 'avg_transaction']
        merchant_stats = merchant_stats.reset_index()
        
        return {
            'top_merchants_by_spending': merchant_stats.nlargest(10, 'total_spent').to_dict('records'),
            'most_frequent_merchants': merchant_stats.nlargest(10, 'transaction_count').to_dict('records'),
            'highest_avg_transaction': merchant_stats.nlargest(5, 'avg_transaction').to_dict('records')
        }
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze spending trends over time"""
        expense_df = df[df['transaction_type'] == 'expense'].copy()
        
        # Monthly spending trends
        monthly_spending = expense_df.groupby(expense_df['date'].dt.to_period('M'))['amount'].sum()
        
        if len(monthly_spending) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Calculate trend
        months = range(len(monthly_spending))
        trend_slope = self._calculate_trend(monthly_spending.values)
        
        # Category trends
        category_trends = {}
        for category in expense_df['category'].unique():
            cat_monthly = expense_df[expense_df['category'] == category].groupby(
                expense_df[expense_df['category'] == category]['date'].dt.to_period('M')
            )['amount'].sum()
            
            if len(cat_monthly) >= 2:
                category_trends[category] = self._calculate_trend(cat_monthly.values)
        
        return {
            'overall_trend': trend_slope,
            'monthly_spending': {str(k): v for k, v in monthly_spending.items()},
            'category_trends': category_trends,
            'trend_interpretation': self._interpret_trend(trend_slope)
        }
    
    def _calculate_trend(self, values: np.array) -> float:
        """Calculate trend slope using linear regression"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1)
        return float(z[0])  # Slope
    
    def _interpret_trend(self, slope: float) -> str:
        """Interpret trend slope"""
        if abs(slope) < 10:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

class BudgetAdvisorAgent(BaseAgent):
    """Agent responsible for budget recommendations and optimization"""
    
    def __init__(self):
        super().__init__("BudgetAdvisor")
        
        # Standard budget allocations (percentage of income)
        self.budget_templates = {
            'conservative': {
                'Housing': 25, 'Food & Dining': 10, 'Transportation': 12,
                'Utilities': 8, 'Healthcare': 5, 'Insurance': 5,
                'Entertainment': 5, 'Shopping': 8, 'Savings': 20, 'Emergency': 2
            },
            'moderate': {
                'Housing': 30, 'Food & Dining': 12, 'Transportation': 15,
                'Utilities': 6, 'Healthcare': 4, 'Insurance': 3,
                'Entertainment': 8, 'Shopping': 10, 'Savings': 15, 'Emergency': 2
            },
            'aggressive': {
                'Housing': 35, 'Food & Dining': 15, 'Transportation': 15,
                'Utilities': 8, 'Healthcare': 5, 'Insurance': 4,
                'Entertainment': 10, 'Shopping': 12, 'Savings': 10, 'Emergency': 1
            }
        }
    
    def generate_budget_recommendations(self, user_data: Dict) -> Dict[str, Any]:
        """Generate personalized budget recommendations"""
        self.log_action("Generating budget recommendations")
        
        monthly_income = user_data.get('monthly_income', 0)
        spending_analysis = user_data.get('spending_analysis', {})
        
        if monthly_income <= 0:
            return {"error": "Monthly income required for budget recommendations"}
        
        # Determine appropriate budget template
        budget_style = self._determine_budget_style(user_data)
        template = self.budget_templates[budget_style]
        
        # Create personalized budget
        recommended_budget = {}
        for category, percentage in template.items():
            recommended_budget[category] = {
                'monthly_limit': round((percentage / 100) * monthly_income, 2),
                'percentage': percentage
            }
        
        # Analyze current vs recommended
        current_analysis = self._analyze_current_vs_recommended(
            spending_analysis, recommended_budget, monthly_income
        )
        
        # Generate specific recommendations
        recommendations = self._generate_specific_recommendations(
            current_analysis, monthly_income
        )
        
        return {
            'budget_style': budget_style,
            'recommended_budget': recommended_budget,
            'current_vs_recommended': current_analysis,
            'recommendations': recommendations,
            'savings_opportunities': self._identify_savings_opportunities(current_analysis)
        }
    
    def _determine_budget_style(self, user_data: Dict) -> str:
        """Determine appropriate budget style based on user profile"""
        monthly_income = user_data.get('monthly_income', 0)
        age = user_data.get('age', 30)
        financial_goals = user_data.get('financial_goals', '').lower()
        risk_tolerance = user_data.get('risk_tolerance', 'moderate').lower()
        
        # Income-based initial classification
        if monthly_income < 4000:
            base_style = 'conservative'
        elif monthly_income > 8000:
            base_style = 'aggressive'
        else:
            base_style = 'moderate'
        
        # Adjust based on other factors
        if 'emergency' in financial_goals or 'conservative' in risk_tolerance:
            if base_style == 'aggressive':
                return 'moderate'
            return 'conservative'
        
        if 'house' in financial_goals and age < 35:
            return 'aggressive'
        
        return base_style
    
    def _analyze_current_vs_recommended(self, spending_analysis: Dict, 
                                      recommended_budget: Dict, 
                                      monthly_income: float) -> Dict:
        """Compare current spending to recommended budget"""
        if 'category_analysis' not in spending_analysis:
            return {"error": "Spending analysis not available"}
        
        current_spending = {}
        category_summary = spending_analysis['category_analysis']['summary']
        
        for category_data in category_summary:
            category = category_data['category']
            current_spending[category] = category_data['total_spent']
        
        comparison = {}
        for category, budget_info in recommended_budget.items():
            current_amount = current_spending.get(category, 0)
            recommended_amount = budget_info['monthly_limit']
            
            comparison[category] = {
                'current_spending': current_amount,
                'recommended_limit': recommended_amount,
                'difference': current_amount - recommended_amount,
                'percentage_of_income': round((current_amount / monthly_income) * 100, 1),
                'status': 'over' if current_amount > recommended_amount else 'under'
            }
        
        return comparison
    
    def _generate_specific_recommendations(self, current_analysis: Dict, 
                                         monthly_income: float) -> List[Dict]:
        """Generate specific, actionable recommendations"""
        recommendations = []
        
        for category, analysis in current_analysis.items():
            if analysis['status'] == 'over':
                difference = analysis['difference']
                percentage_over = (difference / analysis['recommended_limit']) * 100
                
                if percentage_over > 20:  # Significantly over budget
                    recommendations.append({
                        'category': category,
                        'type': 'reduce_spending',
                        'priority': 'high' if percentage_over > 50 else 'medium',
                        'message': f"Consider reducing {category} spending by ${difference:.2f} ({percentage_over:.1f}% over recommended)",
                        'suggested_actions': self._get_category_reduction_tips(category)
                    })
        
        # Savings recommendations
        total_current = sum(analysis['current_spending'] for analysis in current_analysis.values())
        savings_rate = ((monthly_income - total_current) / monthly_income) * 100
        
        if savings_rate < 10:
            recommendations.append({
                'category': 'Savings',
                'type': 'increase_savings',
                'priority': 'high',
                'message': f"Current savings rate is {savings_rate:.1f}%. Aim for at least 10-15%.",
                'suggested_actions': [
                    "Automate savings transfers",
                    "Review and reduce discretionary spending",
                    "Look for additional income sources"
                ]
            })
        
        return sorted(recommendations, key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['priority']], reverse=True)
    
    def _get_category_reduction_tips(self, category: str) -> List[str]:
        """Get category-specific reduction tips"""
        tips = {
            'Food & Dining': [
                "Plan meals and create shopping lists",
                "Cook more meals at home",
                "Use coupons and shop sales",
                "Limit dining out to special occasions"
            ],
            'Entertainment': [
                "Look for free community events",
                "Use streaming services instead of cable",
                "Take advantage of happy hours and discounts",
                "Find free outdoor activities"
            ],
            'Shopping': [
                "Wait 24 hours before non-essential purchases",
                "Compare prices online before buying",
                "Use cashback apps and rewards programs",
                "Buy generic brands when possible"
            ],
            'Transportation': [
                "Use public transportation when available",
                "Carpool or use ride-sharing",
                "Maintain your car to improve efficiency",
                "Consider walking or biking for short trips"
            ],
            'Utilities': [
                "Adjust thermostat settings",
                "Use energy-efficient appliances",
                "Unplug devices when not in use",
                "Compare utility providers"
            ]
        }
        
        return tips.get(category, [
            "Review recent transactions for unnecessary expenses",
            "Look for subscription services to cancel",
            "Negotiate better rates with service providers"
        ])
    
    def _identify_savings_opportunities(self, current_analysis: Dict) -> List[Dict]:
        """Identify specific savings opportunities"""
        opportunities = []
        
        for category, analysis in current_analysis.items():
            if analysis['status'] == 'over':
                potential_savings = min(analysis['difference'], 
                                      analysis['current_spending'] * 0.2)  # Max 20% reduction
                
                if potential_savings > 50:  # Only suggest if savings > $50
                    opportunities.append({
                        'category': category,
                        'potential_monthly_savings': round(potential_savings, 2),
                        'potential_annual_savings': round(potential_savings * 12, 2),
                        'difficulty': self._estimate_reduction_difficulty(category),
                        'impact': 'high' if potential_savings > 200 else 'medium' if potential_savings > 100 else 'low'
                    })
        
        return sorted(opportunities, key=lambda x: x['potential_monthly_savings'], reverse=True)
    
    def _estimate_reduction_difficulty(self, category: str) -> str:
        """Estimate difficulty of reducing spending in category"""
        difficulty_map = {
            'Housing': 'hard',
            'Utilities': 'medium', 
            'Insurance': 'medium',
            'Healthcare': 'hard',
            'Food & Dining': 'easy',
            'Entertainment': 'easy',
            'Shopping': 'easy',
            'Transportation': 'medium',
            'Travel': 'easy'
        }
        
        return difficulty_map.get(category, 'medium')

class MarketContextAgent(BaseAgent):
    """Agent responsible for providing economic context and market awareness"""
    
    def __init__(self):
        super().__init__("MarketContext")
        
    def analyze_market_impact(self, user_data: Dict, market_data: Dict) -> Dict[str, Any]:
        """Analyze how market conditions affect user's financial situation"""
        self.log_action("Analyzing market impact on user finances")
        
        analysis = {
            'inflation_impact': self._analyze_inflation_impact(user_data, market_data),
            'interest_rate_impact': self._analyze_interest_rate_impact(user_data, market_data),
            'employment_context': self._analyze_employment_context(market_data),
            'spending_recommendations': self._generate_market_aware_recommendations(user_data, market_data)
        }
        
        return analysis
    
    def _analyze_inflation_impact(self, user_data: Dict, market_data: Dict) -> Dict:
        """Analyze inflation impact on user's spending"""
        economic_indicators = market_data.get('economic_indicators', {})
        inflation_rate = economic_indicators.get('CPI_All_Items', {}).get('current_value', 3.0)
        
        # Convert to annual percentage if needed (CPI is often an index)
        if inflation_rate > 100:  # Likely CPI index, convert to rate
            inflation_rate = 3.0  # Default assumption
        
        spending_analysis = user_data.get('spending_analysis', {})
        monthly_income = user_data.get('monthly_income', 0)
        
        # Calculate impact on different categories
        category_impacts = {}
        inflation_sensitive_categories = {
            'Food & Dining': 1.2,  # Food inflation often higher
            'Transportation': 1.1,  # Gas prices volatile
            'Utilities': 0.9,      # Utilities somewhat protected
            'Housing': 0.8,        # Housing costs stickier
            'Healthcare': 1.3      # Healthcare inflation often higher
        }
        
        if 'category_analysis' in spending_analysis:
            for category_data in spending_analysis['category_analysis']['summary']:
                category = category_data['category']
                current_spending = category_data['total_spent']
                
                category_inflation = inflation_rate * inflation_sensitive_categories.get(category, 1.0)
                annual_impact = (current_spending * 12) * (category_inflation / 100)
                
                category_impacts[category] = {
                    'monthly_spending': current_spending,
                    'estimated_annual_increase': round(annual_impact, 2),
                    'monthly_increase': round(annual_impact / 12, 2),
                    'category_inflation_rate': round(category_inflation, 1)
                }
        
        total_annual_impact = sum(impact['estimated_annual_increase'] for impact in category_impacts.values())
        
        return {
            'current_inflation_rate': inflation_rate,
            'total_estimated_annual_increase': round(total_annual_impact, 2),
            'monthly_impact': round(total_annual_impact / 12, 2),
            'percentage_of_income': round((total_annual_impact / (monthly_income * 12)) * 100, 1) if monthly_income > 0 else 0,
            'category_impacts': category_impacts,
            'recommendations': self._get_inflation_recommendations(inflation_rate, total_annual_impact, monthly_income)
        }
    
    def _analyze_interest_rate_impact(self, user_data: Dict, market_data: Dict) -> Dict:
        """Analyze interest rate impact on user's finances"""
        economic_indicators = market_data.get('economic_indicators', {})
        fed_rate = economic_indicators.get('Federal_Funds_Rate', {}).get('current_value', 4.5)
        
        # Estimate impact on different aspects
        impacts = {}
        
        # Credit card debt impact (assuming average rate is fed rate + 15%)
        estimated_cc_rate = fed_rate + 15
        impacts['credit_cards'] = {
            'estimated_rate': estimated_cc_rate,
            'recommendation': 'pay_down_debt' if estimated_cc_rate > 20 else 'monitor'
        }
        
        # Savings account impact (assuming high-yield savings)
        estimated_savings_rate = max(0.1, fed_rate - 0.5)  # Usually lower than fed rate
        impacts['savings'] = {
            'estimated_rate': estimated_savings_rate,
            'recommendation': 'shop_for_better_rates' if estimated_savings_rate > 4 else 'consider_alternatives'
        }
        
        # Mortgage impact (rough estimate)
        estimated_mortgage_rate = fed_rate + 2.5
        impacts['mortgage'] = {
            'estimated_rate': estimated_mortgage_rate,
            'recommendation': 'refinance_opportunity' if estimated_mortgage_rate < 6 else 'monitor_market'
        }
        
        return {
            'current_fed_rate': fed_rate,
            'rate_environment': 'high' if fed_rate > 4 else 'low' if fed_rate < 2 else 'moderate',
            'impacts': impacts,
            'general_recommendations': self._get_interest_rate_recommendations(fed_rate)
        }
    
    def _analyze_employment_context(self, market_data: Dict) -> Dict:
        """Analyze employment market context"""
        economic_indicators = market_data.get('economic_indicators', {})
        unemployment_rate = economic_indicators.get('Unemployment_Rate', {}).get('current_value', 4.0)
        
        context = {
            'unemployment_rate': unemployment_rate,
            'employment_environment': 'strong' if unemployment_rate < 4 else 'weak' if unemployment_rate > 6 else 'stable',
            'implications': []
        }
        
        if unemployment_rate < 4:
            context['implications'] = [
                "Strong job market may provide opportunities for career advancement",
                "Good time to negotiate salary increases",
                "Consider building skills for future opportunities"
            ]
        elif unemployment_rate > 6:
            context['implications'] = [
                "Challenging job market - focus on building emergency fund",
                "Consider upskilling to remain competitive",
                "Review job security and have backup plans"
            ]
        else:
            context['implications'] = [
                "Stable employment environment",
                "Good time for career planning and skill development"
            ]
        
        return context
    
    def _generate_market_aware_recommendations(self, user_data: Dict, market_data: Dict) -> List[str]:
        """Generate recommendations based on market conditions"""
        recommendations = []
        
        economic_indicators = market_data.get('economic_indicators', {})
        market_sentiment = market_data.get('market_sentiment', {})
        
        # Inflation-based recommendations
        inflation_rate = economic_indicators.get('CPI_All_Items', {}).get('current_value', 3.0)
        if inflation_rate > 100:
            inflation_rate = 3.0
        
        if inflation_rate > 4:
            recommendations.extend([
                "Consider adjusting budget for higher prices in food and transportation",
                "Look for ways to reduce discretionary spending",
                "Review subscription services for potential savings"
            ])
        
        # Interest rate recommendations
        fed_rate = economic_indicators.get('Federal_Funds_Rate', {}).get('current_value', 4.5)
        if fed_rate > 4:
            recommendations.extend([
                "Prioritize paying down high-interest debt",
                "Shop for high-yield savings accounts",
                "Consider locking in fixed rates for loans"
            ])
        
        # Sentiment-based recommendations
        sentiment = market_sentiment.get('sentiment_description', 'neutral').lower()
        if sentiment == 'negative':
            recommendations.extend([
                "Focus on building emergency fund during uncertain times",
                "Avoid major financial decisions until market stabilizes"
            ])
        elif sentiment == 'positive':
            recommendations.extend([
                "Good time to review and optimize investment allocations",
                "Consider pursuing financial goals while conditions are favorable"
            ])
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _get_inflation_recommendations(self, inflation_rate: float, annual_impact: float, monthly_income: float) -> List[str]:
        """Get specific inflation-related recommendations"""
        recommendations = []
        
        if inflation_rate > 4:
            recommendations.extend([
                "Budget an extra ${:.0f} annually for increased living costs".format(annual_impact),
                "Focus on needs vs wants to combat rising prices",
                "Look for generic brands and bulk purchasing opportunities"
            ])
        
        if annual_impact > monthly_income * 2:  # Impact > 2 months income
            recommendations.append("Consider increasing income or significantly reducing expenses")
        
        return recommendations
    
    def _get_interest_rate_recommendations(self, fed_rate: float) -> List[str]:
        """Get interest rate specific recommendations"""
        if fed_rate > 4:
            return [
                "High interest rate environment - prioritize debt payoff",
                "Take advantage of higher savings rates",
                "Be cautious about taking on new debt"
            ]
        elif fed_rate < 2:
            return [
                "Low interest rate environment - good time for borrowing",
                "Consider refinancing existing debt",
                "Look for better investment opportunities than savings accounts"
            ]
        else:
            return [
                "Moderate interest rate environment",
                "Balance debt payoff with other financial goals",
                "Review all financial products for competitiveness"
            ]

# Multi-Agent Coordinator
class FinancialAgentCoordinator:
    """Coordinates multiple AI agents to provide comprehensive financial analysis"""
    
    def __init__(self):
        self.spending_analyzer = SpendingAnalyzerAgent()
        self.budget_advisor = BudgetAdvisorAgent()
        self.market_context = MarketContextAgent()
        
        self.logger = logging.getLogger("AgentCoordinator")
        self.logger.setLevel(logging.INFO)
    
    def comprehensive_analysis(self, user_data: Dict, market_data: Dict = None) -> Dict[str, Any]:
        """Run comprehensive analysis using all agents"""
        self.logger.info("Starting comprehensive financial analysis")
        
        results = {}
        
        # Run spending analysis
        if 'transactions' in user_data:
            results['spending_analysis'] = self.spending_analyzer.analyze_spending_patterns(
                user_data['transactions']
            )
            # Add spending analysis back to user_data for other agents
            user_data['spending_analysis'] = results['spending_analysis']
        
        # Run budget analysis
        results['budget_recommendations'] = self.budget_advisor.generate_budget_recommendations(user_data)
        
        # Run market context analysis if market data available
        if market_data:
            results['market_context'] = self.market_context.analyze_market_impact(user_data, market_data)
        
        # Generate summary insights
        results['summary'] = self._generate_summary_insights(results)
        
        self.logger.info("Completed comprehensive financial analysis")
        return results
    
    def _generate_summary_insights(self, analysis_results: Dict) -> Dict:
        """Generate high-level summary insights"""
        insights = {
            'top_priorities': [],
            'key_metrics': {},
            'overall_health_score': 0
        }
        
        # Extract key metrics
        if 'spending_analysis' in analysis_results:
            spending = analysis_results['spending_analysis']
            if 'category_analysis' in spending:
                total_spending = sum(cat['total_spent'] for cat in spending['category_analysis']['summary'])
                insights['key_metrics']['monthly_spending'] = total_spending
        
        if 'budget_recommendations' in analysis_results:
            budget = analysis_results['budget_recommendations']
            if 'recommendations' in budget:
                high_priority_recs = [r for r in budget['recommendations'] if r['priority'] == 'high']
                insights['top_priorities'].extend([r['message'] for r in high_priority_recs[:3]])
        
        if 'market_context' in analysis_results:
            market = analysis_results['market_context']
            if 'spending_recommendations' in market:
                insights['top_priorities'].extend(market['spending_recommendations'][:2])
        
        # Calculate basic health score (0-100)
        health_score = 70  # Base score
        
        # Adjust based on budget adherence
        if 'budget_recommendations' in analysis_results:
            budget = analysis_results['budget_recommendations']
            if 'current_vs_recommended' in budget:
                over_budget_categories = sum(1 for cat in budget['current_vs_recommended'].values() 
                                           if cat['status'] == 'over')
                total_categories = len(budget['current_vs_recommended'])
                if total_categories > 0:
                    budget_adherence = 1 - (over_budget_categories / total_categories)
                    health_score += (budget_adherence - 0.5) * 40  # Adjust by up to ±20 points
        
        insights['overall_health_score'] = max(0, min(100, int(health_score)))
        insights['top_priorities'] = insights['top_priorities'][:5]  # Limit to top 5
        
        return insights

# Usage example and testing
if __name__ == "__main__":
    # Sample transaction data
    sample_transactions = pd.DataFrame([
        {'date': '2024-09-01', 'amount': 1200, 'category': 'Housing', 'merchant': 'Landlord', 'transaction_type': 'expense'},
        {'date': '2024-09-02', 'amount': 45, 'category': 'Food & Dining', 'merchant': 'Grocery Store', 'transaction_type': 'expense'},
        {'date': '2024-09-03', 'amount': 5000, 'category': 'Salary', 'merchant': 'Company', 'transaction_type': 'income'},
        {'date': '2024-09-04', 'amount': 85, 'category': 'Food & Dining', 'merchant': 'Restaurant', 'transaction_type': 'expense'},
        {'date': '2024-09-05', 'amount': 500, 'category': 'Shopping', 'merchant': 'Amazon', 'transaction_type': 'expense'},
    ])
    
    # Sample user data
    sample_user_data = {
        'monthly_income': 5000,
        'age': 30,
        'financial_goals': 'Emergency Fund',
        'risk_tolerance': 'Moderate',
        'transactions': sample_transactions
    }
    
    # Sample market data
    sample_market_data = {
        'economic_indicators': {
            'CPI_All_Items': {'current_value': 3.2},
            'Federal_Funds_Rate': {'current_value': 4.5},
            'Unemployment_Rate': {'current_value': 3.8}
        },
        'market_sentiment': {
            'sentiment_description': 'Neutral',
            'overall_sentiment': 0.1
        }
    }
    
    # Test individual agents
    coordinator = FinancialAgentCoordinator()
    
    # Run comprehensive analysis
    results = coordinator.comprehensive_analysis(sample_user_data, sample_market_data)
    
    print("=== COMPREHENSIVE FINANCIAL ANALYSIS ===")
    print(f"Overall Health Score: {results['summary']['overall_health_score']}/100")
    print("\nTop Priorities:")
    for priority in results['summary']['top_priorities']:
        print(f"- {priority}")
    
    print(f"\nMonthly Spending: ${results['summary']['key_metrics'].get('monthly_spending', 0)}")
    
    if 'budget_recommendations' in results:
        print(f"\nBudget Style: {results['budget_recommendations']['budget_style']}")
        print("Savings Opportunities:")
        for opp in results['budget_recommendations'].get('savings_opportunities', [])[:3]:
            print(f"- {opp['category']}: ${opp['potential_monthly_savings']}/month")
    
    if 'market_context' in results:
        inflation_impact = results['market_context']['inflation_impact']
        print(f"\nInflation Impact: +${inflation_impact['monthly_impact']}/month")
        
        interest_impact = results['market_context']['interest_rate_impact']
        print(f"Interest Rate Environment: {interest_impact['rate_environment']}")