#importing necessary libraries
import pandas as pd
import numpy as np
from flask import Flask,render_template,flash,request,flash, session
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import mysql.connector


db=mysql.connector.connect(user="root",password="",port='3306',database='hate_speech')
cur=db.cursor()


app=Flask(__name__)

app.secret_key="CBJcb786874wrf78chdchsdcv"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/job')
def job():
    return render_template('job.html')


@app.route('/login',methods=['POST','GET'])
def login():
    if request.method=='POST':
        useremail=request.form['useremail']
        session['useremail']=useremail
        userpassword=request.form['userpassword']
        sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
        cur.execute(sql)
        data=cur.fetchall()
        db.commit()
        if data ==[]:
            msg="user Credentials Are not valid"
            return render_template("login.html",name=msg)
        else:
            return render_template("userhome.html",myname=data[0][1])
    return render_template('login.html')

@app.route('/registration',methods=["POST","GET"])
def registration():
    if request.method=='POST':
        username=request.form['username']
        useremail = request.form['useremail']
        userpassword = request.form['userpassword']
        conpassword = request.form['conpassword']
        Age = request.form['Age']
        
        contact = request.form['contact']
        if userpassword == conpassword:
            sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
            cur.execute(sql)
            data=cur.fetchall()
            db.commit()
            print(data)
            if data==[]:
                
                sql = "insert into user(Name,Email,Password,Age,Mob)values(%s,%s,%s,%s,%s)"
                val=(username,useremail,userpassword,Age,contact)
                cur.execute(sql,val)
                db.commit()
                flash("Registered successfully","success")
                return render_template("login.html")
            else:
                flash("Details are invalid","warning")
                return render_template("registration.html")
        else:
            flash("Password doesn't match", "warning")
            return render_template("registration.html")
    return render_template('registration.html')

@app.route('/load',methods=["GET","POST"])
def load():
    global df, dataset
    if request.method == "POST":
        data = request.files['data']
        df = pd.read_csv(data)
        dataset = df.head(100)
        msg = 'Data Loaded Successfully'
        return render_template('load.html', msg=msg)
    return render_template('load.html')

@app.route('/view')
def view():
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())

def preprocess_data(df):
    
    # Convert text to lowercase
    df['text'] = df['text'].str.strip().str.lower()
    return df

@app.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global x, y, x_train, x_test, y_train, y_test, countvectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100

        print('#########################################')

        df = pd.read_csv("data job posts.csv")
        
        # Check for and drop NaN values
        df = df.dropna(subset=['JobDescription', 'IT'])

        # Check the size of your dataset
        print("Size of the dataset: ", df.shape[0])

        df = df[['JobDescription', 'IT']]
        df.head()
        x = df['JobDescription']
        y = df['IT']

        # Check if there is enough data for training
        if len(df) == 0:
            return render_template('preprocess.html', msg='Error: Insufficient data for training set')

        # Adjust the split ratio if needed
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=42)

        # Check the size of the training set after splitting
        print("Size of the training set: ", len(x_train))

        if len(x_train) == 0:
            return render_template('preprocess.html', msg='Error: Insufficient data for training set')

        vec = CountVectorizer(stop_words='english')

        x_train = vec.fit_transform(x_train).toarray()
        x_test = vec.transform(x_test).toarray()

        # Describes info about the train and test set
        print("Number transactions X_train dataset: ", x_train.shape)
        print("Number transactions y_train dataset: ", y_train.shape)
        print("Number transactions X_test dataset: ", x_test.shape)
        print("Number transactions y_test dataset: ", y_test.shape)

        print(x_train, x_test)

        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')

    return render_template('preprocess.html')


@app.route('/model',methods=['POST','GET'])
def model():

    if request.method=="POST":
        print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
        s=int(request.form['algo'])
        if s==0:
            return render_template('model.html',msg='Please Choose an Algorithm to Train')
        elif s==1:
            import pickle
            # filename = 'models/Random_forest1.sav'
            # model = pickle.load(open(filename, 'rb'))
            model = RandomForestClassifier()
            model.fit(x_train,y_train)
            pred = model.predict(x_test)
            acc_rf = accuracy_score(y_test,pred)
            
           
            msg = 'The accuracy obtained by Random Forest Classifier is ' + str(acc_rf) + str('%')
            return render_template('model.html', msg=msg)
        elif s==2:
            model1 = DecisionTreeClassifier()
            model1.fit(x_train,y_train)
            pred1 = model1.predict(x_test)
            acc_dt = accuracy_score(y_test,pred1)
            
            msg = 'The accuracy obtained by DecisionTreeClassifier is ' + str(acc_dt) + str('%')
            return render_template('model.html', msg=msg)
        elif s==3:
            
            model2 = XGBClassifier()
            model2.fit(x_train,y_train)
            pred2 = model2.predict(x_test)
            acc_xgb = accuracy_score(y_test,pred2)
            
            msg = 'The accuracy obtained by XGBoost Classifier is ' + str(acc_xgb) + str('%')
            return render_template('model.html', msg=msg)
        elif s==4:
            
            model3 = LogisticRegression()
            model3.fit(x_train,y_train)
            pred3 = model3.predict(x_test)
            acc_lr = accuracy_score(y_test,pred3)
            
            msg = 'The accuracy obtained by Logistic regression is ' + str(acc_lr) + str('%')
            return render_template('model.html', msg=msg)
        
    return render_template('model.html')

import pickle

@app.route('/prediction',methods=['POST','GET'])
def prediction():
    global x_train,y_train
    if request.method == "POST":
        f1 = request.form['text']
        print(f1)
        
        filename='models/DecisionTreeClassifier1.sav'
        model = pickle.load(open(filename, 'rb'))
        from sklearn.feature_extraction.text import HashingVectorizer
        hvectorizer = HashingVectorizer(n_features=52991,norm=None,alternate_sign=False)
        from sklearn.feature_extraction.text import CountVectorizer
        vec = CountVectorizer(stop_words='english')

        result =model.predict(hvectorizer.transform([f1]))
        result=result[0]
        print(result)
        if result==0:
            msg = 'The Job Post is Genuine'
        elif result==1:
            msg= 'This is a fake job post'
               
        return render_template('prediction.html',msg=msg)    

    return render_template('prediction.html')







if __name__=='__main__':
    app.run(debug=True)