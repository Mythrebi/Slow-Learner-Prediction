# -*- coding: utf-8 -*-
"""
Created on Tue Mar 8 09:38:37 2019

@author: Mythrebi Selvan
"""
from django.shortcuts import render
import csv
import pandas as pa
import numpy as np
import re

def algorithms(sample_test): 
    result=[]
    
    #############SVM
    #Import svm model
    
    from sklearn import svm
    clf = svm.SVC(kernel='linear') # Linear Kernel
    clf.fit(X_train, y_train)
    #y_pred = clf.predict(y_test)
    y_pred1 = clf.predict(sample_test)
    result.append(y_pred1[0])
    print(y_pred[0])
    y_pred = classifier.predict(y_test)
    from sklearn.metrics import classification_report, confusion_matrix  
    print(confusion_matrix(y_test, y_pred))  
    print(classification_report(y_test, y_pred)) 
    
    
    #####################knn
    
    from sklearn.neighbors import KNeighborsClassifier  
    classifier = KNeighborsClassifier(n_neighbors=15)  
    classifier.fit(X_train, y_train)  
    y_pred2 = classifier.predict(sample_test)
    result.append(y_pred2[0]) 
    y_pred = classifier.predict(y_test)
    from sklearn.metrics import classification_report, confusion_matrix  
    print(confusion_matrix(y_test, y_pred))  
    print(classification_report(y_test, y_pred))
    
    ############
    from sklearn.naive_bayes import GaussianNB
    nb=GaussianNB()
    nb.fit(X_train,y_train)
    y_pred3=nb.predict(sample_test)
    y_pred = classifier.predict(y_test)
    from sklearn.metrics import classification_report, confusion_matrix  
    print(confusion_matrix(y_test, y_pred))  
    print(classification_report(y_test, y_pred))
    
    result.append(y_pred3[0])
    rescount=[0,0,0]  
    for k in range(0,len(result)):
        ind=result[k]
        rescount[ind-1]=rescount[ind-1]+1
    got=rescount.index(max(rescount)) 
    return got+1


def classifystudents(request):
    filename = 'G:\\MYTHU MATERIALS\\8th sem\\cloud\\project\\Educational_data.csv'
    def cleangender(s):
        if(s=="M"):
            t=1
        else:
            if(s=="F"):
                t=2
            else:
                t=s
        return t
    def cleanstage(s):
        if(s=="lowerlevel"):
            t=1
        else:
            if(s=="MiddleSchool"):
                t=2
            else:
                if(s=="HighSchool"):
                    t=3
                else:
                    t=s
        return t
    
    def cleanabsence(s):
        if(s=="Under-7"):
            t=1
        else:
            if(s=="Above-7"):
                t=2
            else:
                t=s
        return t
    
    def cleanclass(s):
        if(s=="L"):
            t=1
        else:
            if(s=="M"):
                t=2
            else:
                if(s=="H"):
                    t=3
                else:
                    t=s
        return t
    
    
    filename = 'G:\\MYTHU MATERIALS\\8th sem\\cloud\\project\\Educational_data.csv'
    fields = [] 
    rows = [] 
    
    with open(filename, 'r') as csvfile: 
        csvreader = csv.reader(csvfile) 
    	
        for row in csvreader:
            row[0]=cleangender(row[0])
            row[1]=cleanstage(row[1])
            row[6]=cleanabsence(row[6])
            row[7]=cleanclass(row[7])
            rows.append(row) 
    
    print("Total no. of rows: %d"%(csvreader.line_num)) 
    
    print('Field names are:' + ', '.join(field for field in fields)) 
     
    xt=pa.DataFrame(rows)
    xt
    xt.to_csv('G:\\MYTHU MATERIALS\\8th sem\\cloud\\project\\res.csv',index=False,header=False)
    
    #preprocess
    filename='G:\\MYTHU MATERIALS\\8th sem\\cloud\\project\\res.csv'
    dataset = pa.read_csv(filename)
    
    # Load the Diabetes dataset
    df=pa.DataFrame(dataset)
    df=df.drop('Class',1)
    y = dataset.Class # define the target variable (dependent variable) as y
    from sklearn.model_selection import train_test_split  
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    
    
    ############input data
    #2	1	42	30	13	70	2(medium)
    #2	2	100	80	95	90	1(high)
    #1	2	19	65	50	10	2(low)
    if request.POST:
        gender=request.POST.get('Gender', '')
        StageID=request.POST.get('StageID','')
        raisedhands=request.POST.get('Raisehands','')
        VisITedResources=request.POST.get('Visited Resources','')
        AnnouncementsView=request.POST.get('Announcements','')
        Discussion=request.POST.get('Discussion','')
        StudentAbsenceDays==request.POST.get('Absentdays','')
        gen=cleangender(gender)
        stage=cleanstage(StageID)
        absence=cleanabsence(StudentAbsenceDays)
        getval=[]
        inp=[]
        getval.append(gen)
        getval.append(stage)
        getval.append(raisedhands)
        getval.append(VisITedResources)
        getval.append(AnnouncementsView)
        getval.append(Discussion)
        getval.append(absence)
        inp.append(getval)
        sample_test=pa.DataFrame(inp)
        t=algorithms(sample_test)
        print(t)
    render(request, 'index.html', {'resp': t})