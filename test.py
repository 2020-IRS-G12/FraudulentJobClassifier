from job_projects import lr,be,ga_job
import pandas as pd

df1 = pd.read_csv('data/data_lr.csv')
df1 = df1.head()
df1.drop([1,2,3,4],inplace=True)

df2 =  pd.read_csv('data/data_bert.csv')
df2 = df2.head()
df2.drop([1,2,3,4],inplace=True)

possi_lr = lr(df1)
possi_b = be(df2)
job_last = ga_job(possi_lr,possi_b)
print(job_last)