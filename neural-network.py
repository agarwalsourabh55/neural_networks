#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np 
import pandas as pd 



# In[ ]:




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

geography=pd.get_dummies(X["Geography"],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)


X=pd.concat([X,geography,gender],axis=1)

X=X.drop(['Geography','Gender'],axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)





# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[ ]:




def model(input_layer, hidden_layer,intializer, initializer_output, activation, activation_output,optimiz,loss,drop):
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 12, kernel_initializer = initializer,activation=activation,input_dim =input_layer))
    if drop==True:
        classifier.add(Dropout(0.3))
    
    
    # Adding the second hidden layer
    for  i in range(2,hidden_layer+1):
        classifier.add(Dense(units = 12, kernel_initializer = initializer,activation=activation))
        if drop==True:
            classifier.add(Dropout(0.3))

        
    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = initializer_output, activation = activation_output))

    # Compiling the NN
    classifier.compile(optimizer = optimiz , loss = loss, metrics = ['accuracy'])
    print(classifier.summary())

    # Fitting the NN to the Training set
    model_history=classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 100, epochs = 100)
    return model_history,classifier


def visualize(model_history):
    print(model_history.history.keys())


    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def predict(classifier ,thres,X_test,y_test):

    y_pred = classifier.predict(X_test)
    y_pred=np.where(y_pred>thres,1,0)
    
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Calculate the Accuracy
    score=accuracy_score(y_pred,y_test)
    return score,cm



# In[ ]:


initializer=keras.initializers.HeNormal()
initializer_output= keras.initializers.GlorotNormal()

input_layer =11
hidden_layer = 3

activation='relu'
activation_output='sigmoid'

optimiz= 'adam'
loss=keras.losses.BinaryCrossentropy()

model_output=model(input_layer,hidden_layer ,  initializer, initializer_output,activation, activation_output,optimiz,loss ,drop= False)

visualize(model_output)
predict(classifier,0.5,X_test,y_test)


# CAN SEE CLEARLY THAT THIS IS OVERFITTING PROBLEM
# 
# So to remove this I will use drop_out
# 

# In[ ]:


initializer=keras.initializers.HeNormal()
initializer_output= keras.initializers.GlorotNormal()

input_layer =11
hidden_layer = 3

activation='relu'
activation_output='sigmoid'

optimiz= 'adam'
loss=keras.losses.BinaryCrossentropy()

model_output,classifier=model(input_layer,hidden_layer ,  initializer, initializer_output,activation, activation_output,optimiz,loss ,drop= True)

visualize(model_output)
predict(classifier,0.5,X_test,y_test)

