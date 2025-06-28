"""
Constants and patterns for hiring-related analysis
"""

# Semantic queries for hiring analysis (broader corporate language)
HIRING_QUERIES = [
    "employees and workforce management",
    "talent acquisition and human resources",
    "labor costs and employee compensation", 
    "recruiting and staffing operations",
    "employee turnover and retention",
    "workforce development and training",
    "competition for talent and skilled workers",
    "human capital and personnel",
    "employment and labor market",
    "staffing levels and workforce planning"
]

# More specific queries for unique challenges
SPECIFIC_HIRING_QUERIES = [
    "difficulty hiring qualified employees",
    "shortage of skilled workers and technicians",
    "increased labor costs and wage inflation",
    "employee turnover rates and retention issues",
    "competition for specialized talent",
    "remote work and workforce flexibility challenges",
    "unionization and labor negotiations",
    "skills gap and training requirements",
    "geographic constraints on hiring",
    "regulatory compliance and workforce management"
]

# Query categories for different types of analysis
QUERY_CATEGORIES = {
    'general': HIRING_QUERIES,
    'specific': SPECIFIC_HIRING_QUERIES,
    'all': HIRING_QUERIES + SPECIFIC_HIRING_QUERIES
}

# Comparison queries for company comparison analysis
COMPARISON_QUERIES = [
    "human capital and personnel",
    "employees and workforce management", 
    "talent acquisition and human resources"
] 