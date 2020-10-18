from flask import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from werkzeug.utils import redirect
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from job_projects import lr,be,ga_job,fliter
import sys

import pandas as pd
import numpy as np

app = Flask(__name__)

data = pd.read_csv('data/data_lr.csv')

@app.route('/')
def start():
    return render_template('start.html')

@app.route('/form')
def formOne():
    return render_template('form.html')

@app.route('/result', methods=['POST','GET'])
def showTheResult():

    bert_text = str(request.form['title'])+" "+str(request.form['department'])+" "+str(request.form['company_profile'])+" "+\
    str(request.form['description'])+" "+str(request.form['requirements'])+" "+str(request.form['benefits'])+" "+str(request.form['employment_type'])+" "+\
    str(request.form['required_experience'])+" "+str(request.form['required_education'])+" "+str(request.form['industry'])+" "+str(request.form['function'])+" "
    # if len(bert_text)<=20:
    #     bert_text = 'nan'
    bert_data =  pd.DataFrame([bert_text],columns=['text'])

    temp = []
    lr_text = str(request.form['title'])+" "+str(request.form['department'])+" "+str(request.form['company_profile'])+" "+\
    str(request.form['description'])+" "+str(request.form['requirements'])+" "+str(request.form['benefits'])
    # if len(lr_text)<=20:
    #     lr_text = 'nan'
    temp.append(lr_text)
    temp.append(request.form['telecommuting'])
    temp.append(request.form['has_company_logo'])
    temp.append(request.form['has_questions'])
    temp.append(request.form['employment_type'])
    temp.append(request.form['required_experience'])
    temp.append(request.form['required_education'])
    temp.append(request.form['industry'])
    temp.append(request.form['function'])
    lr_data = pd.DataFrame([temp],columns=data.columns.values)
    
    if len(bert_text)==0:
        job_last = 1
    elif len(bert_text)<= 20:
        job_last = 1
    elif fliter(bert_data):
        possi_lr = lr(lr_data)
        possi_b = be(bert_data)
        job_last = ga_job(possi_lr,possi_b)
    else:
        job_last = 1

    result_text = ''
    if(job_last==0):
        result_text = 'True'
    else:
        result_text = 'Fraudulent'

    return render_template('result.html',result = result_text ) 


app.run(debug=False)
