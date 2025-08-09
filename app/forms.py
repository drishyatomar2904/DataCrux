from flask_wtf import FlaskForm
from wtforms import SelectField, TextAreaField, SubmitField
from wtforms.validators import DataRequired, InputRequired
from flask_wtf.file import FileField, FileRequired, FileAllowed

class UploadForm(FlaskForm):
    file = FileField('Dataset File', validators=[
        FileRequired(message="Please select a file"),
        FileAllowed(['csv', 'xlsx', 'xls'], 'CSV or Excel files only!')
    ])
    submit = SubmitField('Upload')

class CleanForm(FlaskForm):
    missing = SelectField('Handle Missing Values', choices=[
        ('drop', 'Drop rows with missing values'),
        ('mean', 'Fill with mean (numerical)'),
        ('median', 'Fill with median (numerical)'),
        ('mode', 'Fill with mode (categorical)')
    ], default='mean', validators=[InputRequired()])
    
    duplicates = SelectField('Handle Duplicates', choices=[
        ('drop', 'Drop duplicate rows'),
        ('keep', 'Keep duplicates')
    ], default='drop', validators=[InputRequired()])
    
    outliers = SelectField('Handle Outliers', choices=[
        ('remove', 'Remove outliers'),
        ('keep', 'Keep outliers')
    ], default='remove', validators=[InputRequired()])
    
    submit = SubmitField('Clean Data')

class PredictForm(FlaskForm):
    target = SelectField(
        'Target Column', 
        choices=[],  # Will be populated dynamically
        validators=[DataRequired(message="Please select a target column")]
    )
    
    problem_type = SelectField(
        'Problem Type', 
        choices=[
            ('classification', 'Classification'),
            ('regression', 'Regression'),
            ('clustering', 'Clustering')
        ], 
        default='classification',
        validators=[InputRequired()]
    )
    
    submit = SubmitField('Run Prediction')

class QuestionForm(FlaskForm):
    question = TextAreaField('Ask a question about your data', 
                             validators=[DataRequired(message="Please enter a question")])
    submit = SubmitField('Get Answer')