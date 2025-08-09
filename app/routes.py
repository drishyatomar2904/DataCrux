from flask import Blueprint, render_template, request, redirect, url_for, flash, session, send_file, current_app
import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from app.forms import UploadForm, CleanForm, PredictForm, QuestionForm  # Fixed import path
from app.utils import clean_data, generate_insights, predict_churn, generate_report, ask_groq

bp = Blueprint('main', __name__)

@bp.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@bp.route('/upload', methods=['GET', 'POST'])
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        
        # Get absolute upload path
        upload_folder = current_app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)  # Ensure directory exists
        filepath = os.path.join(upload_folder, filename)
        
        try:
            file.save(filepath)
        except Exception as e:
            current_app.logger.error(f"Error saving file: {e}")
            flash('Error saving file. Please try again.', 'danger')
            return redirect(url_for('main.upload'))
        
        # Read data
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath)
            else:
                flash('Unsupported file format', 'danger')
                return redirect(url_for('main.upload'))
        except Exception as e:
            current_app.logger.error(f"Error reading file: {e}")
            flash('Error reading file. Please check the file format.', 'danger')
            return redirect(url_for('main.upload'))
        
        # Store in session
        session['filepath'] = filepath
        session['filename'] = filename
        session['df'] = df.to_json()
        session['columns'] = list(df.columns)
        
        flash('File uploaded successfully!', 'success')
        return redirect(url_for('main.dashboard'))
    
    return render_template('upload.html', form=form)

@bp.route('/dashboard', methods=['GET'])
def dashboard():
    if 'df' not in session:
        flash('Please upload a file first', 'warning')
        return redirect(url_for('main.upload'))
    
    df = pd.read_json(session['df'])
    summary = df.describe().to_html()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return render_template('dashboard.html', 
                          summary=summary, 
                          columns=session['columns'],
                          num_cols=num_cols,
                          data_info={'rows': df.shape[0], 'columns': df.shape[1]})

@bp.route('/clean', methods=['GET', 'POST'])
def clean():
    form = CleanForm()
    if 'df' not in session:
        flash('Please upload a file first', 'warning')
        return redirect(url_for('main.upload'))
    
    df = pd.read_json(session['df'])
    
    if form.validate_on_submit():
        cleaned_df = clean_data(df, form.missing.data, form.duplicates.data, form.outliers.data)
        session['df'] = cleaned_df.to_json()
        session['columns'] = list(cleaned_df.columns)
        flash('Data cleaned successfully!', 'success')
        return redirect(url_for('main.dashboard'))
    
    return render_template('clean.html', form=form, columns=session['columns'])

@bp.route('/insights', methods=['GET'])
def insights():
    if 'df' not in session:
        flash('Please upload a file first', 'warning')
        return redirect(url_for('main.upload'))
    
    df = pd.read_json(session['df'])
    graphs = generate_insights(df)
    return render_template('insights.html', graphs=graphs)

@bp.route('/predict', methods=['GET', 'POST'])
def predict():
    form = PredictForm()
    if 'df' not in session:
        flash('Please upload a file first', 'warning')
        return redirect(url_for('main.upload'))
    
    df = pd.read_json(session['df'])
    
    # Populate target column choices dynamically
    form.target.choices = [(col, col) for col in df.columns]
    
    if form.validate_on_submit():
        target = form.target.data
        problem_type = form.problem_type.data
        results = predict_churn(df, target, problem_type)
        return render_template('predictions.html', results=results)
    
    return render_template('predict.html', form=form)

@bp.route('/ask', methods=['GET', 'POST'])
def ask():
    form = QuestionForm()
    if 'df' not in session:
        flash('Please upload a file first', 'warning')
        return redirect(url_for('main.upload'))
    
    df = pd.read_json(session['df'])
    
    if form.validate_on_submit():
        question = form.question.data
        answer = ask_groq(df, question)
        return render_template('ask.html', form=form, answer=answer)
    
    return render_template('ask.html', form=form)

@bp.route('/report', methods=['GET'])
def report():
    if 'df' not in session:
        flash('Please upload a file first', 'warning')
        return redirect(url_for('main.upload'))
    
    df = pd.read_json(session['df'])
    report_path = generate_report(df)
    return send_file(report_path, as_attachment=True)

@bp.route('/clear_session', methods=['GET'])
def clear_session():
    session.clear()
    flash('Session cleared. You can upload a new file.', 'info')
    return redirect(url_for('main.upload'))