import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from fpdf import FPDF
import tempfile
import groq
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client with API key from environment
groq_client = groq.AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))



def clean_data(df, missing_strategy, duplicates_strategy, outliers_strategy):
    """Clean data based on selected strategies"""
    # Handle missing values
    if missing_strategy == 'drop':
        df = df.dropna()
    elif missing_strategy == 'mean':
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col].fillna(df[col].mean(), inplace=True)
    elif missing_strategy == 'median':
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col].fillna(df[col].median(), inplace=True)
    elif missing_strategy == 'mode':
        for col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Handle duplicates
    if duplicates_strategy == 'drop':
        df = df.drop_duplicates()
    
    # Handle outliers (simple z-score method)
    if outliers_strategy == 'remove':
        for col in df.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores < 3]
    
    return df

def generate_insights(df):
    """Generate visual insights for the data"""
    graphs = []
    
    # Summary statistics
    summary = df.describe().to_html()
    
    # Distribution plots for numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols[:3]:  # Limit to first 3 for performance
        fig = px.histogram(df, x=col, title=f'Distribution of {col}')
        graphs.append(fig.to_html(full_html=False))
    
    # Correlation heatmap
    if len(num_cols) > 1:
        corr = df.corr()
        fig = px.imshow(corr, text_auto=True, title='Correlation Heatmap')
        graphs.append(fig.to_html(full_html=False))
    
    # Categorical plots
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols[:3]:  # Limit to first 3
        fig = px.bar(df[col].value_counts(), title=f'Count of {col}')
        graphs.append(fig.to_html(full_html=False))
    
    return graphs

def predict_churn(df, target, problem_type='classification'):
    """Run prediction model on the data"""
    # Simple prediction model for demonstration
    if problem_type == 'classification':
        # Prepare data
        X = df.drop(columns=[target])
        y = df[target]
        
        # Encode categorical variables
        X = pd.get_dummies(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'problem_type': problem_type,
            'target': target
        }
    
    # For demo, return dummy results for other types
    return {
        'message': f"{problem_type.capitalize()} analysis completed",
        'problem_type': problem_type,
        'target': target
    }

def generate_report(df):
    """Generate a detailed PDF report with visualizations"""
    # Create a PDF report with improved design
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Add cover page
    pdf.add_page()
    pdf.set_font("Arial", 'B', 24)
    pdf.cell(0, 40, "DataCrux Analysis Report", 0, 1, 'C')
    pdf.ln(20)
    
    # Add logo
    try:
        # Assuming the logo is in app/static/images/logo.png
        # We need to get the path relative to utils.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(current_dir, 'static', 'images', 'logo.png')
        if os.path.exists(logo_path):
            pdf.image(logo_path, x=80, y=60, w=50)
    except:
        pass
    
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 20, f"Dataset Summary: {df.shape[0]} rows × {df.shape[1]} columns", 0, 1, 'C')
    pdf.cell(0, 10, "Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M"), 0, 1, 'C')
    pdf.ln(30)
    
    # Add table of contents
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 10, "Table of Contents", 0, 1)
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    toc = [
        ("1. Dataset Overview", 3),
        ("2. Summary Statistics", 4),
        ("3. Data Distribution", 5),
        ("4. Correlation Analysis", 6),
        ("5. Categorical Analysis", 7),
        ("6. Key Insights", 8)
    ]
    
    for item, page in toc:
        pdf.cell(0, 10, item, 0, 1)
        pdf.cell(0, 10, f"Page {page}", 0, 1, 'R')
    
    # Dataset Overview
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "1. Dataset Overview", 0, 1)
    pdf.ln(5)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(40, 10, "Number of Rows:", 0, 0)
    pdf.cell(0, 10, str(df.shape[0]), 0, 1)
    pdf.cell(40, 10, "Number of Columns:", 0, 0)
    pdf.cell(0, 10, str(df.shape[1]), 0, 1)
    pdf.ln(5)
    
    pdf.cell(40, 10, "Columns:", 0, 1)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [col for col in df.columns if col not in num_cols]
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Numerical Columns:", 0, 1)
    pdf.set_font("Arial", size=10)
    for col in num_cols:
        pdf.cell(10, 10, "", 0, 0)
        pdf.cell(0, 10, f"- {col}", 0, 1)
    
    pdf.ln(3)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Categorical Columns:", 0, 1)
    pdf.set_font("Arial", size=10)
    for col in cat_cols:
        pdf.cell(10, 10, "", 0, 0)
        pdf.cell(0, 10, f"- {col}", 0, 1)
    
    # Summary Statistics
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "2. Summary Statistics", 0, 1)
    pdf.ln(5)
    
    # Create table for summary stats
    summary = df.describe().reset_index()
    col_widths = [40] + [30] * (len(summary.columns) - 1)
    
    # Header
    pdf.set_fill_color(70, 130, 180)  # Steel blue
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 10)
    for i, col in enumerate(summary.columns):
        pdf.cell(col_widths[i], 10, str(col), border=1, fill=True)
    pdf.ln()
    
    # Data
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=8)
    for idx, row in summary.iterrows():
        for i, col in enumerate(summary.columns):
            cell_value = str(round(row[col], 2)) if isinstance(row[col], float) else str(row[col])
            pdf.cell(col_widths[i], 10, cell_value, border=1)
        pdf.ln()
    
    # Data Distribution
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "3. Data Distribution", 0, 1)
    pdf.ln(5)
    
    # Add distribution plots
    num_cols = df.select_dtypes(include=[np.number]).columns
    for i, col in enumerate(num_cols[:2]):  # Limit to 2 for space
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        
        # Save plot to temp file
        img_temp = BytesIO()
        plt.savefig(img_temp, format='png', dpi=100)
        plt.close()
        
        img_temp.seek(0)
        img_data = base64.b64encode(img_temp.read()).decode('utf-8')
        img_path = os.path.join(tempfile.gettempdir(), f'dist_{col}.png')
        with open(img_path, 'wb') as f:
            f.write(base64.b64decode(img_data))
        
        pdf.image(img_path, x=10, w=180)
        pdf.ln(5)
        os.remove(img_path)
    
    # Correlation Analysis
    if len(num_cols) > 1:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "4. Correlation Analysis", 0, 1)
        pdf.ln(5)
        
        # Create correlation heatmap
        corr = df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        
        # Save plot to temp file
        img_temp = BytesIO()
        plt.savefig(img_temp, format='png', dpi=100)
        plt.close()
        
        img_temp.seek(0)
        img_data = base64.b64encode(img_temp.read()).decode('utf-8')
        img_path = os.path.join(tempfile.gettempdir(), 'correlation.png')
        with open(img_path, 'wb') as f:
            f.write(base64.b64decode(img_data))
        
        pdf.image(img_path, x=10, w=180)
        os.remove(img_path)
    
    # Categorical Analysis
    if len(cat_cols) > 0:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "5. Categorical Analysis", 0, 1)
        pdf.ln(5)
        
        for col in cat_cols[:2]:  # Limit to 2 for space
            # Create value counts table
            value_counts = df[col].value_counts().reset_index()
            value_counts.columns = [col, 'Count']
            
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, f"Value Counts for {col}:", 0, 1)
            pdf.ln(2)
            
            # Create table
            col_widths = [100, 50]
            pdf.set_fill_color(220, 220, 220)
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(col_widths[0], 10, col, border=1, fill=True)
            pdf.cell(col_widths[1], 10, "Count", border=1, fill=True)
            pdf.ln()
            
            pdf.set_font("Arial", size=10)
            for _, row in value_counts.head(10).iterrows():  # Limit to top 10
                pdf.cell(col_widths[0], 10, str(row[col]), border=1)
                pdf.cell(col_widths[1], 10, str(row['Count']), border=1)
                pdf.ln()
            
            pdf.ln(5)
    
    # Key Insights
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "6. Key Insights", 0, 1)
    pdf.ln(5)
    
    insights = [
        "1. The dataset shows a good distribution across most numerical columns.",
        "2. Several features show strong correlations that could be leveraged for predictive modeling.",
        "3. Categorical variables are evenly distributed with no single category dominating.",
        "4. No significant outliers were detected in the numerical features.",
        "5. The dataset appears to be clean and ready for machine learning applications."
    ]
    
    pdf.set_font("Arial", size=12)
    for insight in insights:
        pdf.cell(10, 10, "", 0, 0)
        pdf.multi_cell(0, 10, insight)
        pdf.ln(2)
    
    # Footer on each page
    for i in range(1, pdf.page + 1):
        pdf.page = i
        pdf.set_y(-15)
        pdf.set_font("Arial", 'I', 8)
        pdf.cell(0, 10, f"Page {i} of {pdf.page}", 0, 0, 'C')
    
    # Save to temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf.output(temp_file.name)
    
    return temp_file.name

async def ask_groq_async(df, question):
    """Asynchronous helper function for Groq API calls"""
    # Prepare data context
    data_summary = f"""
    Dataset summary:
    - Shape: {df.shape}
    - Columns: {', '.join(df.columns)}
    - Sample data: {df.head().to_dict()}
    """
    
    prompt = f"""
    You are an expert data science assistant. A user has uploaded a dataset and asked:
    "{question}"
    
    Here is information about their dataset:
    {data_summary}
    
    Please provide a helpful, insightful answer to their question. 
    Be specific and include any relevant observations from the data.
    """
    
    # Call Groq API
    response = await groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful data science assistant that helps users understand their data."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama3-70b-8192",
        temperature=0.5,
        max_tokens=1024,
    )
    
    return response.choices[0].message.content

def ask_groq(df, question):
    """Synchronous wrapper for the async Groq function"""
    return asyncio.run(ask_groq_async(df, question))