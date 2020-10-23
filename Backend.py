from flask import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from werkzeug.utils import html, redirect
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from job_projects import lr,be,ga_job,fliter
import sys

import HighlightSentence as hs

import pandas as pd
import numpy as np
import nltk

#nltk.download('punkt')
#nltk.download('stopwords')


app = Flask(__name__)

data = pd.read_csv('data/data_lr.csv')

class TextInfo:
    word_list = []
    company_profile = ''
    requirements = ''
    benefits = ''
    description = ''

text_info = TextInfo()

@app.route('/')
def start():
    return render_template('start.html')

@app.route('/form')
def formOne():
    return render_template('form.html')

@app.route('/result', methods=['POST','GET'])
def showTheResult():
    mform = {}
    for k in request.form:
        if request.form[k] == "":
            mform[k] = "nan"
        else:
            mform[k] = request.form[k]

    #import pdb; pdb.set_trace()

    bert_text = str(mform['title'])+" "+str(mform['department'])+" "+str(mform['company_profile'])+" "+\
    str(mform['description'])+" "+str(mform['requirements'])+" "+str(mform['benefits'])+" "+str(mform['employment_type'])+" "+\
    str(mform['required_experience'])+" "+str(mform['required_education'])+" "+str(mform['industry'])+" "+str(mform['function'])+" "

    text_info.company_profile = str(mform['company_profile'])
    text_info.requirements = str(mform['requirements'])
    text_info.benefits = str(mform['benefits'])
    text_info.description = str(mform['description'])

    # if len(bert_text)<=20:
    #     bert_text = 'nan'
    bert_data =  pd.DataFrame([bert_text],columns=['text'])

    temp = []
    lr_text = str(mform['title'])+" "+str(mform['department'])+" "+str(mform['company_profile'])+" "+\
    str(mform['description'])+" "+str(mform['requirements'])+" "+str(mform['benefits'])
    # if len(lr_text)<=20:
    #     lr_text = 'nan'
    temp.append(lr_text)
    temp.append(mform['telecommuting'])
    temp.append(mform['has_company_logo'])
    temp.append(mform['has_questions'])
    temp.append(mform['employment_type'])
    temp.append(mform['required_experience'])
    temp.append(mform['required_education'])
    temp.append(mform['industry'])
    temp.append(mform['function'])
    lr_data = pd.DataFrame([temp],columns=data.columns.values)

    if len(bert_text)==0:
        pjob = [0,1]
        job_last = 1
    elif len(bert_text)<= 50:
        pjob = [0,1]
        job_last = 1
    elif fliter(bert_data):
        possi_lr = lr(lr_data)
        possi_b,word_list = be(bert_data)
        print(possi_lr)
        print(possi_b)
        text_info.word_list = word_list
        pjob,job_last = ga_job(possi_lr,possi_b)
    else:
        pjob = [0,1]
        job_last = 1

    result_text = ''
    possibility = '0%'
    if(job_last==0):
        result_text = 'Credible'
        possibility = str(round(pjob[0],3)*100) + "%"
    else:
        result_text = 'Suspicious'
        possibility = str(round(pjob[1],3)*100) + "%"

    return render_template('result.html',result = result_text, possibility=possibility,isFraudulent = job_last)

@app.route('/report', methods=['POST','GET'])
def showTheReport():

    html_text = hs.highlightSentence('Company Profile',text_info.company_profile,text_info.word_list)
    html_text = html_text+hs.highlightSentence('Requirements',text_info.requirements,text_info.word_list)
    html_text = html_text+hs.highlightSentence('Benefits',text_info.benefits,text_info.word_list)
    html_text = html_text+hs.highlightSentence('Description',text_info.description,text_info.word_list)

    return render_template('report.html',html_text=html_text)

app.run(debug=False)
