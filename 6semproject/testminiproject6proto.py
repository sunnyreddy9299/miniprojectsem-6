#import tensorflow as t
import tensorflow as tf
import numpy as np
np.random.seed(200)
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
import pandas as pd
#from tensorflow import set_random_seed

#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from keras.layers import BatchNormalization
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#from keras.models import Model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
def preprocess(url1,url2):
    dataframe=pd.read_csv(url1)
    d=pd.read_csv(url2)
    x_train=dataframe[['Temp in F','Humidity']]
    #print(x_train)
    y_train=dataframe[['rainfall','weather condition']]
    #print(y_train)
    x_test=d[['Temp in F','Humidity']]
    #print(x_test)
    y_test=d[['rainfall','weather condition']]
    #print(y_test)
    #min_max_scaler =StandardScaler()
    #x_train= min_max_scaler.fit_transform(x_train)
    y_train=y_train.drop('weather condition',axis=1)
    y_test=y_test.drop('weather condition',axis=1)
    x_train=normalize(x_train)
    x_test=normalize(x_test)
    y_train=normalize(y_train)
    y_test=normalize(y_test)
    '''
    print("X_TRAIN IS:")
    print(x_train)
    '''
    #x_test=min_max_scaler.fit_transform(x_test)
    '''
    print("X_TEST IS:")
    print(x_test)
    '''
    #print(y_test1)
    test=dataframe[['weather condition']].replace('Fair',0)
    test1=test[['weather condition']].replace('Haze',1)
    test2=test1[['weather condition']].replace('Fog',2)
    test3=test2[['weather condition']].replace('partly cloudy',3)
    test4=test3[['weather condition']].replace('Mostly cloudy',4)
    test51=test4[['weather condition']].replace('Rain',5)
    y_train1=test51
    '''
    print("TEST is:")
    print(test4)
    '''
    #y_train1=y_train.drop('weather condition',axis=1)
    #print(y_train1)
    '''
    print("Y_TRAIN IS:")
    print(y_train1)
    '''
    test=d[['weather condition']].replace('Fair',0)
    test1=test[['weather condition']].replace('Haze',1)
    test2=test1[['weather condition']].replace('Fog',2)
    test3=test2[['weather condition']].replace('partly cloudy',3)
    test4=test3[['weather condition']].replace('Mostly cloudy',4)
    test5=test4[['weather condition']].replace('Rain',5)
    y_test1=test5
    print(x_train)
    print(y_train)
    print(x_test)
    print(y_test)
    print(y_train1)
    print(y_test1)
    return x_train,y_train,x_test,y_test,y_train1,y_test1

def modeltrain1(X_train,Y_train):
    model=Sequential()
    #input layer)
    model.add(Dense(units=2,use_bias=True))
    #model.add(Dense(2))
    #hidden layers
    model.add(BatchNormalization())
    model.add(LSTM(units=2,return_sequences=True,use_bias=True))
    #model.add(Dropout(0.4))
    model.add(Dense(units=128,use_bias=True))
    model.add(LSTM(units=128,return_sequences=True,use_bias=True))
    #model.add(Dropout(0.4))
    model.add(Dense(units=32,use_bias=True))
    model.add(LSTM(units=32,return_sequences=True,use_bias=True))
    #model.add(Dropout(0.4))
    model.add(Dense(128,use_bias=True))
    model.add(LSTM(units=128,return_sequences=True,use_bias=True))
    model.add(Dense(units=32,use_bias=True))
    model.add(LSTM(units=32,return_sequences=True,use_bias=True))
    #model.add(Dropout(0.4))
    #output layer
    model.add(Dense(units=1,use_bias=True))
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['acc'],verbose=1)
    model.fit(X_train,Y_train,epochs=100,verbose=2)
    print(model.summary())
    return model
def modeltrainweather1(X_train,Y_train1):
    model=Sequential()
    #input layer
    model.add(Dense(units=2,use_bias=True))
    model.add(Dense(2))
    #hidden layers
    model.add(BatchNormalization())
    model.add(LSTM(units=2,return_sequences=True,use_bias=True))
    model.add(Dense(units=128,use_bias=True))
    #model.add(LSTM(units=128,return_sequences=True,use_bias=True))
    model.add(Dense(units=128,use_bias=True))
    #model.add(LSTM(units=64,return_sequences=True,use_bias=True))
    model.add(Dense(128,use_bias=True))
    #model.add(LSTM(units=128,return_sequences=True,use_bias=True))
    model.add(Dense(units=32,use_bias=True))
    #model.add(LSTM(units=32,return_sequences=True,use_bias=True))
    model.add(Dense(6,use_bias=True))
    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['binary_accuracy'],verbose=1)
    model.fit(X_train,Y_train1,epochs=100,verbose=2)
    print(model.summary())
    return model
def testandeval(model,X_test,Y_test):
    Model=model
    print(X_test)
    pred=Model.predict(X_test,verbose=1)
    print(pred)
    Y_test=np.reshape(Y_test,(Y_test.shape[0]))
    print(Y_test.shape)
    pred=np.reshape(pred,(pred.shape[0]))
    print(Y_test)
    print(pred)
    score=r2_score(pred,Y_test)
    print("r2_score:",score)
    mae=mean_absolute_error(pred,Y_test)
    print("Mean absolute error:",mae)
    mse=mean_squared_error(pred,Y_test)
    print("Mean squared error:",mse)

def modeleval(model,X_test,Y_test):
    pred1=model.predict_classes(X_test,verbose=1)
    #pred1=pred1.argmax(axis=-1)
    pred1=np.reshape(pred1,(pred1.shape[0]))
    Y_test=np.reshape(Y_test,(Y_test.shape[0]))
    print(Y_test.shape)
    pred1=np.reshape(pred1,(pred1.shape[0]))
    print(Y_test)
    print(pred1)
    acc=confusion_matrix(pred1,Y_test)
    res=accuracy_score(pred1,Y_test)
    print("confusion matrix:",acc)
    print("accuracy score:",res)
    
np.random.seed(200)
url1='C:\\Users\\Yadav\\Desktop\\mini project folder\\train.csv'
url2='C:\\Users\\Yadav\\Desktop\\mini project folder\\test.csv'
#np.random.seed(1)
a,b,c,d,c1,d1=preprocess(url1,url2)
a=np.array(a)
b=np.array(b)
c1=np.array(c1)
d1=np.array(d1)
c=np.array(c)
d=np.array(d)
a=np.reshape(a,(a.shape[0], 1, a.shape[1]))
print(a)
b=np.reshape(b,(b.shape[0], 1, b.shape[1]))
print(b)
c1=np.reshape(c1,(c1.shape[0],1,1))
print(c1)
print(c.shape)
c=np.reshape(c,(c.shape[0],1,c.shape[1]))
print(c)
d1=np.reshape(d1,(d1.shape[0],1,1))
d=np.reshape(d,(d.shape[0],1,1))
c1=to_categorical(c1,num_classes=6)
#d1=to_categorical(d1,num_classes=6)

print("--------------------------------------------------------------------")
print("Rainfall prediction............................................")
model123=modeltrain1(a,b)
testandeval(model123,c,d)
print("Weather prediction............................................")
M=modeltrainweather1(a,c1)
modeleval(M,c,d1)