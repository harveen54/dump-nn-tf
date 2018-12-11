
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# ### Step 1: Importing Data

# In[2]:


df= pd.read_csv('train.csv')


# In[3]:


df.head()


# ### Step 2: Cleaning Data

# In[4]:


df.info()


# In[5]:


df.isnull().values.any()


# In[6]:


df.isnull().sum()


# In[7]:


df[df['V4'].isnull()]


# In[8]:


df.dropna(inplace=True)


# In[9]:


df.isnull().values.any()


# ### Step 3: Data Preprocessing

# In[10]:


df.describe()


# In[11]:


df.groupby('Class').count()


# In[12]:


X_df = df.drop(columns=['Class'])
y_df = df['Class']


# In[13]:


X_df.shape, y_df.shape


# In[14]:


X = np.array(X_df)
y = np.array(y_df)


# In[15]:


from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(sparse=False)
y_new = one.fit_transform(y.reshape(y.shape[0],1))
y_new.shape


# In[16]:


y = y_new


# In[17]:


# #optional step
# from sklearn.preprocessing import MinMaxScaler
# scale = MinMaxScaler()
# scale.fit_transform(X)


# ### Step 4: Splitting Data into Train and Valid Set

# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.33, shuffle=True)


# In[20]:


X_train.shape , X_valid.shape, y_train.shape, y_valid.shape


# In[21]:


from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
X_train = scale.fit_transform(X_train)
X_valid = scale.transform(X_valid)


# ### Step 5: Modelling Data

# In[22]:


import tensorflow as tf


# In[23]:


def model(x):
    mu = 0
    sigma = 0.05
    
    
    
    weights_1 = tf.Variable(tf.truncated_normal(shape=(30,1000) ,mean = mu, stddev= sigma))
    bias_1 = tf.Variable(tf.zeros(1000))
    product_1 = tf.matmul(x, weights_1 ) + bias_1
    
    layer_1 = tf.nn.relu(product_1)
    
    weights_2 = tf.Variable(tf.truncated_normal(shape=(1000,500) , mean= mu, stddev=sigma))
    bias_2 = tf.Variable(tf.zeros(500))
    product_2 = tf.matmul(layer_1 , weights_2  ) + bias_2 

    layer_2 = tf.nn.relu(product_2)

    weights_3 = tf.Variable(tf.truncated_normal(shape=(500,50) , mean= mu, stddev=sigma))
    bias_3 = tf.Variable(tf.zeros(50))
    product_3 = tf.matmul( layer_2 , weights_3  ) + bias_3
    
    layer_3= tf.nn.relu(product_3)
    
    weights_4 = tf.Variable(tf.truncated_normal(shape=(50,1) , mean= mu, stddev=sigma))
    bias_4 = tf.Variable(tf.zeros(1))
    product_4 = tf.matmul( layer_3 , weights_4  ) + bias_4

    
    return product_4
    


# In[24]:


def new_model(x):
    #l1 = tf.layers.batch_normalization(x)
    l1 = tf.layers.dense(x, 1000,  activation='relu', kernel_initializer='glorot_uniform')
    #l1 = tf.layers.dropout(l1, rate=0.2)
    
    #l2 = tf.layers.batch_normalization(l1)
    l2 = tf.layers.dense(l1,500, activation='relu', kernel_initializer='glorot_uniform' )
    
    #l3 = tf.layers.batch_normalization(l2)
    l3 = tf.layers.dense(l2, 50,  activation='relu', kernel_initializer='glorot_uniform')
    
    #l4 = tf.layers.batch_normalization(l3)
    l4 = tf.layers.dense(l3, 2,kernel_initializer='glorot_uniform' )
    
    return l4


# In[25]:


# forward propagation 
# model(x)

learn_rate = 0.0001

x_pass = tf.placeholder(dtype=tf.float32,  shape = (None, 30))
y_pass = tf.placeholder(dtype = tf.int32 , shape = (None,2))

forward_prop =  new_model(x_pass)

loss = tf.nn.softmax_cross_entropy_with_logits(labels = y_pass,logits=forward_prop)

#loss = tf.keras.backend.binary_crossentropy(target=y_pass, output=forward_prop)

total_loss = tf.reduce_mean(loss)

optimizer = tf.train.AdamOptimizer(learning_rate = learn_rate)

backward_prop = optimizer.minimize(total_loss)


# In[26]:


epochs = 1000


# In[27]:


#correct_prediction = tf.equal(tf.cast( tf.greater_equal(loss,0.5) , tf.int32) , y_pass)
#accuracy_operation =  tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[28]:


correct_prediction = tf.equal(tf.argmax(forward_prop,1) ,tf.argmax (y_pass,1))
accuracy_operation =  tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[29]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Training...')
    for i in range(epochs):
        sess.run( backward_prop , feed_dict={x_pass: X_train , y_pass: y_train})
        
        train_loss = sess.run(total_loss, feed_dict={x_pass:X_train , y_pass: y_train})
        validation_loss = sess.run(total_loss, feed_dict={x_pass:X_valid , y_pass: y_valid})
        train_acc =  sess.run(accuracy_operation, feed_dict={x_pass:X_train, y_pass:y_train})
        accuracy = sess.run(accuracy_operation, feed_dict={x_pass:X_valid, y_pass:y_valid})
        print(train_loss,validation_loss, train_acc, accuracy)
    


# In[30]:


import keras
from keras.layers import BatchNormalization
from keras.layers.core import Flatten,Dense,Dropout, Activation, Lambda
from keras.models import Model, Sequential
from keras.optimizers import Adam,SGD
from keras.callbacks import TensorBoard
.


# In[ ]:


model = Sequential()
#model.add(Lambda(lambda x: x , input_shape = (24)))
#model.add(Lambda(lambda x: x+0.1 , input_shape = (64,64,3)))
#model.add(Flatten())
Dense(1000)
model.add(BatchNormalization(input_shape=(30,)))
model.add(Dense(1000))
model.add(Activation('relu'))


model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(50))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(optimizer = Adam(lr=0.001) , loss = 'binary_crossentropy', metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


model.fit(X_train, y_train, epochs=30, verbose =1, validation_data=(X_valid, y_valid))


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(model.history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()


# In[ ]:


plt.plot(model.history.history['acc'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Curve')
plt.show()


# ### Step 6: Inference on Test Set

# In[ ]:


test_df = pd.read_csv('./test.csv')


# In[ ]:


test_df.head()


# In[ ]:


y_test = np.array(test_df['Class'])


# In[ ]:


x_test =  np.array(test_df.drop(columns='Class'))


# In[ ]:


x_test.shape, y_test.shape


# ### Step 7: Checking Metrics for Performance

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


y_pred = model.predict(x_test)


# In[ ]:


y_pred = (y_pred >= 0.5).astype(int)


# In[ ]:


y_pred[0:5]


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


print(confusion_matrix(y_test,y_pred))

