import pandas as pd
import numpy as np 
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("creditcard.csv")

df.loc[df.Class == 0, 'Not_Fraud'] = 1
df.loc[df.Class == 1, 'Not_Fraud'] = 0

df['Amount_max_fraud'] = 1
df.loc[df.Amount <= 2125.87, 'Amount_max_fraud'] = 0

df['V1_'] = df.V1.map(lambda x: 1 if x < -3 else 0)
df['V2_'] = df.V2.map(lambda x: 1 if x > 2.5 else 0)
df['V3_'] = df.V3.map(lambda x: 1 if x < -4 else 0)
df['V4_'] = df.V4.map(lambda x: 1 if x > 2.5 else 0)
df['V5_'] = df.V5.map(lambda x: 1 if x < -4.5 else 0)
df['V6_'] = df.V6.map(lambda x: 1 if x < -2.5 else 0)
df['V7_'] = df.V7.map(lambda x: 1 if x < -3 else 0)
df['V9_'] = df.V9.map(lambda x: 1 if x < -2 else 0)
df['V10_'] = df.V10.map(lambda x: 1 if x < -2.5 else 0)
df['V11_'] = df.V11.map(lambda x: 1 if x > 2 else 0)
df['V12_'] = df.V12.map(lambda x: 1 if x < -2 else 0)
df['V14_'] = df.V14.map(lambda x: 1 if x < -2.5 else 0)
df['V16_'] = df.V16.map(lambda x: 1 if x < -2 else 0)
df['V17_'] = df.V17.map(lambda x: 1 if x < -2 else 0)
df['V18_'] = df.V18.map(lambda x: 1 if x < -2 else 0)
df['V19_'] = df.V19.map(lambda x: 1 if x > 1.5 else 0)
df['V21_'] = df.V21.map(lambda x: 1 if x > 0.6 else 0)

df = df.rename(columns={'Class': 'Fraud'})

#Create dataframes of only Fraud and Not_Fraud transactions.
Fraud = df[df.Fraud == 1]
Not_Fraud = df[df.Not_Fraud == 1]

# X_train has 80% of the fraudulent transactions.
X_train = Fraud.sample(frac=0.8)
count_Frauds = len(X_train)

X_train = pd.concat([X_train, Not_Fraud.sample(frac = 0.8)], axis = 0)

X_test = df.loc[~df.index.isin(X_train.index)]

# shuffle data
X_train = shuffle(X_train)
X_test = shuffle(X_test)

# Add our target features to y_train and y_test.
y_train = X_train.Fraud
y_train = pd.concat([y_train, X_train.Not_Fraud], axis=1)

y_test = X_test.Fraud
y_test = pd.concat([y_test, X_test.Not_Fraud], axis=1)


X_train = X_train.drop(['Fraud','Not_Fraud'], axis = 1)
X_test = X_test.drop(['Fraud','Not_Fraud'], axis = 1)

ratio = len(X_train)/count_Frauds 

y_train.Fraud *= ratio
y_test.Fraud *= ratio

df.head()

mean = df.drop(['Fraud','Not_Fraud'], axis=1).mean()
std = df.drop(['Fraud','Not_Fraud'], axis=1).std()

X_train=(X_train-mean)/std
X_test =(X_test-mean)/std

# Split the testing data into validation and testing sets
split = int(len(y_test)/2)

inputX = X_train.to_numpy()
inputY = y_train.to_numpy()
inputX_valid = X_test.to_numpy()[:split]
inputY_valid = y_test.to_numpy()[:split]
inputX_test = X_test.to_numpy()[split:]
inputY_test = y_test.to_numpy()[split:]

X_train.head()

# Number of input nodes.
input_nodes = 48 

# nodes in each hidden layer
hidden_nodes1 = 40
hidden_nodes2 = 30
hidden_nodes3 = 20
hidden_nodes4 = 10

# Percent of nodes to keep during dropout.
pkeep = tf.placeholder(tf.float32)

# input
x = tf.placeholder(tf.float32, [None, input_nodes])

# layer 1
W1 = tf.Variable(tf.truncated_normal([input_nodes, hidden_nodes1], stddev = 0.15))
b1 = tf.Variable(tf.zeros([hidden_nodes1]))
y1 = tf.nn.relu(tf.matmul(x, W1) + b1)
# y1 = tf.nn.dropout(y1, rate=1-pkeep)

# layer 2
W2 = tf.Variable(tf.truncated_normal([hidden_nodes1, hidden_nodes2], stddev = 0.15))
b2 = tf.Variable(tf.zeros([hidden_nodes2]))
y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)
# y2 = tf.nn.dropout(y2, rate=1-pkeep)

# layer 3
W3 = tf.Variable(tf.truncated_normal([hidden_nodes2, hidden_nodes3], stddev = 0.15)) 
b3 = tf.Variable(tf.zeros([hidden_nodes3]))
y3 = tf.nn.relu(tf.matmul(y2, W3) + b3)
# y3 = tf.nn.dropout(y3, rate=1-pkeep)

# layer 4
W4 = tf.Variable(tf.truncated_normal([hidden_nodes3, hidden_nodes4], stddev = 0.15)) 
b4 = tf.Variable(tf.zeros([hidden_nodes4]))
y4 = tf.nn.relu(tf.matmul(y3, W4) + b4)
y4 = tf.nn.dropout(y4, rate=1-pkeep)

# layer 5
W5 = tf.Variable(tf.truncated_normal([hidden_nodes4, 2], stddev = 0.15)) 
b5 = tf.Variable(tf.zeros([2]))
y5 = tf.nn.softmax(tf.matmul(y4, W5) + b5)

# output
y = y5
y_ = tf.placeholder(tf.float32, [None, 2])

# Parameters
training_epochs = 1000
training_dropout = 0.9
display_step = 1 
n_samples = y_train.shape[0]
batch_size = 2048
regularizer_rate = 0.1
learning_rate = 0.0005

# Cost function: Cross Entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y,labels=y_)) \
+ regularizer_rate*(tf.reduce_sum(tf.square(b1)))

# We will optimize our model via AdamOptimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Correct prediction if the most likely value (Fraud or Not_Fraud) from softmax equals the target value.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

accuracy_summary = []
cost_summary = []
valid_accuracy_summary = [] 
valid_cost_summary = [] 
stop_early = 0 

# Save the best weights so that they can be used to make the final predictions
checkpoint = "best_model.ckpt"
saver = tf.train.Saver(max_to_keep=1)

# Initialize variables and tensorflow session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epochs): 
        for batch in range(int(n_samples/batch_size)):
            batch_x = inputX[batch*batch_size : (1+batch)*batch_size]
            batch_y = inputY[batch*batch_size : (1+batch)*batch_size]

            sess.run([optimizer], feed_dict={x: batch_x, 
                                             y_: batch_y,
                                             pkeep: training_dropout})

        # Display logs after every 10 epochs
        if (epoch) % display_step == 0:
            train_accuracy, newCost = sess.run([accuracy, cost], feed_dict={x: inputX, 
                                                                            y_: inputY,
                                                                            pkeep: training_dropout})

            valid_accuracy, valid_newCost = sess.run([accuracy, cost], feed_dict={x: inputX_valid, 
                                                                                  y_: inputY_valid,
                                                                                  pkeep: 1})

            print ("Epoch:", epoch,
                   "Acc =", "{:.5f}".format(train_accuracy), 
                   "Cost =", "{:.5f}".format(newCost),
                   "Valid_Acc =", "{:.5f}".format(valid_accuracy), 
                   "Valid_Cost = ", "{:.5f}".format(valid_newCost))
            
            # Save the weights if these conditions are met.
            if epoch > 0 and valid_accuracy > max(valid_accuracy_summary) and valid_accuracy > 0.995:
                saver.save(sess, checkpoint)
            
            # Record the results of the model
            accuracy_summary.append(train_accuracy)
            cost_summary.append(newCost)
            valid_accuracy_summary.append(valid_accuracy)
            valid_cost_summary.append(valid_newCost)
            
            # If the model does not improve after 15 logs, stop the training.
            if valid_accuracy < max(valid_accuracy_summary) and epoch > 50:
                stop_early += 1
                if stop_early == 15:
                    break
            else:
                stop_early = 0
            

    print("Optimization Complete")