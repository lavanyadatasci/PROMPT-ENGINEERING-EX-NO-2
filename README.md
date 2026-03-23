# EX-02-Cross-Platform-Prompting-Evaluating-Diverse-Techniques-in-AI-Powered-Text-Summarization

## AIM
To evaluate and compare the effectiveness of prompting techniques (zero-shot, few-shot, chain-of-thought, role-based) across different AI platforms (e.g., ChatGPT, Gemini, Claude, Copilot) in a specific task: text summarization.

## SCENARIO:
You are part of a content curation team for an educational platform that delivers quick summaries of research papers to undergraduate students. Your task is to summarize a 500-word technical article on "The Basics of Blockchain Technology" using multiple AI platforms and prompting strategies.

Your goal is to determine which combination of prompting technique + platform provides the best summary in terms of:

Accuracy

Coherence

Simplicity

Speed

User experience

## OUTPUT
import pandas as pd
df=pd.read_csv(r"C:\Users\Downloads\titanic_dataset.csv")
df
<img width="1310" height="504" alt="image" src="https://github.com/user-attachments/assets/55f72a21-8a2c-4735-8e84-85408a2d212c" />
df.shape
<img width="1209" height="33" alt="image" src="https://github.com/user-attachments/assets/16459f5f-2764-459f-a8f4-9a7ceedc296f" />
df.set_index("PassengerId",inplace=True)
df
<img width="1351" height="542" alt="image" src="https://github.com/user-attachments/assets/9a51e367-6127-4e43-9c14-22b7ac81584b" />
df.nunique
<img width="1115" height="829" alt="image" src="https://github.com/user-attachments/assets/f957e136-df25-459c-b6c5-f6fa3d691954" />
df['Sex'].value_counts()
<img width="1225" height="87" alt="image" src="https://github.com/user-attachments/assets/e30f72f0-1aed-4dfb-a833-74365d97c600" />
df.Survived.unique()
<img width="1209" height="43" alt="image" src="https://github.com/user-attachments/assets/2f72f429-8402-4f6d-9a25-18a2acd1cca7" />
df.rename(columns={"Sex":"Gender"},inplace=True)
df
<img width="1326" height="515" alt="image" src="https://github.com/user-attachments/assets/b29dcf6b-337c-4152-b784-56ba96673196" />
import seaborn as sns
sns.countplot(data=df)
<img width="1108" height="588" alt="image" src="https://github.com/user-attachments/assets/12a51bf3-16d7-43a9-a609-bd4c172d5051" />
sns.countplot(x="Survived",hue="Gender",data=df)
<img width="1110" height="572" alt="image" src="https://github.com/user-attachments/assets/7973951e-c4b7-4459-844e-8049485e10e2" />
sns.catplot(x="Survived",hue="Gender",data=df,kind="count")
<img width="1131" height="643" alt="image" src="https://github.com/user-attachments/assets/3877280b-00f9-4e73-8300-f59cdafa2de7" />
sns.catplot(x="Survived",hue="Gender",data=df,kind="violin")
<img width="1132" height="637" alt="image" src="https://github.com/user-attachments/assets/d45881af-4c4f-4b81-a22a-c7201efebb36" />
sns.boxplot(data=df)
<img width="1264" height="549" alt="image" src="https://github.com/user-attachments/assets/d0257480-cee8-488d-bb89-dc6e29815432" />
df.boxplot(column="Survived",by="Gender")
<img width="1275" height="594" alt="image" src="https://github.com/user-attachments/assets/2173013b-68fe-433c-aad2-b8b2b1a9b89a" />
sns.scatterplot(data=df)
<img width="962" height="545" alt="image" src="https://github.com/user-attachments/assets/f47c622d-ddca-4642-b02e-542ef601b68b" />
sns.scatterplot(x=df['Age'],y=df['Fare'])
<img width="1267" height="552" alt="image" src="https://github.com/user-attachments/assets/168407c4-5745-477f-a9a9-b4e618344794" />
sns.jointplot(x='Age',y='Fare',data=df)
<img width="1256" height="763" alt="image" src="https://github.com/user-attachments/assets/7fda8267-3f5e-47e3-a622-73835a1b3f05" />
sns.jointplot(x='Age',y='Fare',data=df,kind="kde")
<img width="1279" height="762" alt="image" src="https://github.com/user-attachments/assets/0cd48c4c-3742-467f-bcc1-4be7700de646" />
sns.jointplot(x='Age',y='Fare',data=df,kind="hist")
<img width="1030" height="759" alt="image" src="https://github.com/user-attachments/assets/947c24b3-8d8f-4856-97d2-e2b91eed7525" />
sns.catplot(x='Gender',col='Survived',data=df,kind='count',color='green')
sns.pairplot(data=df)
<img width="901" height="927" alt="image" src="https://github.com/user-attachments/assets/65881982-71a3-4830-beab-8b0205a3b894" />
corr1=df.select_dtypes(include=["number"]).corr()
sns.heatmap(corr1,annot=True)
<img width="1066" height="632" alt="image" src="https://github.com/user-attachments/assets/5de9c569-9ac8-41bb-a7fa-54b03668b4e0" />
sns.catplot(x='Gender',col='Survived',data=df,kind='count',hue="Pclass")
<img width="1353" height="640" alt="image" src="https://github.com/user-attachments/assets/fe42a27b-558b-40ae-ac54-f92634411fe9" />
import matplotlib.pyplot as plt
fig,ax1=plt.subplots(figsize=(8,5))
pt=sns.boxplot(ax=ax1,x='Pclass',y='Age',hue='Gender',data=df)
<img width="1159" height="558" alt="image" src="https://github.com/user-attachments/assets/891c8319-dfbf-4393-83a3-577eb3a821c2" />
























## RESULT
Thus performing Exploratory Data Analysis on the given data set.

