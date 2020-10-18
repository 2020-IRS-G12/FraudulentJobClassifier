import pandas as pd

data = pd.read_csv('data/data_17.csv')
data_columns = data.columns.tolist()
data_columns = data_columns[-1:] + data_columns[:-1]
data = data[data_columns]
data = data.drop(['fraudulent','label'],axis=1)
data.to_csv('data_lr.csv',index=False)
data['text'] = list(data['text'].map(str)+' '+data['employment_type'].map(str)+' '+data['required_experience'].map(str)+ \
' '+data['required_education'].map(str)+' '+data['industry'].map(str)+' '+data['function'].map(str))
data = data.drop(['telecommuting','has_company_logo','has_questions','employment_type','required_experience','required_education','industry','function'],axis=1)
data.to_csv('data_bert.csv',index=False)
            