import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split



monday = pd.read_csv('Monday-WorkingHours.pcap_ISCX.csv')
tuesday = pd.read_csv('Tuesday-WorkingHours.pcap_ISCX.csv')
wednesday = pd.read_csv('Wednesday-workingHours.pcap_ISCX.csv')
thursday_morning = pd.read_csv('Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv')
thursday_afternoon = pd.read_csv('Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv')
friday_morning = pd.read_csv('Friday-WorkingHours-Morning.pcap_ISCX.csv')
friday = pd.read_csv('Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')

monday = monday.dropna()
tuesday = tuesday.dropna()
wednesday = wednesday.dropna()
thursday_morning = thursday_morning.dropna()
thursday_afternoon = thursday_afternoon.dropna()
friday_morning = friday_morning.dropna()
friday = friday.dropna()

# clean data
convert_dict = {' Subflow Bwd Packets': int} 
monday = monday.astype(convert_dict)

days = pd.concat([monday, tuesday, wednesday, thursday_morning, thursday_afternoon, friday_morning, friday])


column_names = list(days.columns.values)
processed_names = []
for name in column_names:
    processed_names.append(name.strip())

days.columns = processed_names

convert_dict = {'Flow Packets/s': np.float64,
                'Flow Bytes/s': np.float64} 

days = days.astype(convert_dict)

days = days.replace([np.inf], np.nan).dropna()

days = days.drop(['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 
                  'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate'], axis=1)

days = shuffle(days)

labels = days['Label']
days = days.drop('Label', axis=1)


labels[labels.str.contains('Web Attack')] = 'Web Attack'
labels[labels.str.contains('DoS')] = 'DoS'
labels[labels.str.contains('Heartbleed')] = 'DoS'
labels[labels.str.contains('BENIGN')] = 'Normal'
labels[labels.str.contains('Patator')] = 'Brute Force'


days=(days-days.mean())/(days.std())


X_train, X_test, y_train, y_test = train_test_split(days, labels, test_size = 0.1)


X_train_m = X_train.values
y_train_m = y_train.values

X_test_m = X_test.values
y_test_m = y_test.values

num_classes = y_train_m.shape[1]
n_samples = y_train_m.shape[0]
num_features = X_train_m.shape[1]


num_output = y_train_m.shape[1]
num_layers_0 = 50
regularizer_rate = 0.1


# Placeholders for the input data
input_X = tf.placeholder('float32',shape = (None,num_features), name='input_X')
input_y = tf.placeholder('float32',shape = (None,num_classes), name='input_Y')

# for dropout layer
keep_prob = tf.placeholder(tf.float32)

# Store layers weight & bias
weights = {
    '0': tf.Variable(tf.random_normal([num_features,num_layers_0], stddev=(1/tf.sqrt(float(num_features))))),
    '1': tf.Variable(tf.random_normal([num_layers_0,num_output], stddev=(1/tf.sqrt(float(num_layers_0))))),
}

biases = {
    '0': tf.Variable(tf.random_normal([num_layers_0])),
    '1': tf.Variable(tf.random_normal([num_output])),
}

# Initializing weights and biases
hidden_output_0 = tf.nn.relu(tf.matmul(input_X,weights['0'])+biases['0'])
hidden_output_0_0 = tf.nn.dropout(hidden_output_0, rate = 1-keep_prob)

predicted_y = tf.sigmoid(tf.matmul(hidden_output_0_0,weights['1']) + biases['1'])


# Defining the loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted_y,labels=input_y)) \
+ regularizer_rate*(tf.reduce_sum(tf.square(biases['0'])))


learning_rate = 0.1
# Adam optimizer for finding the right weight

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,var_list=[weights['0'], weights['1'],
                                                                         biases['0'], biases['1']])

# Metrics definition
correct_prediction = tf.equal(tf.argmax(y_train,1), tf.argmax(predicted_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Training parameters
batch_size = 100
epochs = 100
dropout = 0.9


training_accuracy = []
training_loss = []
testing_accuracy = []

s = tf.InteractiveSession()

s.run(tf.global_variables_initializer())
for epoch in range(epochs):
    for batch in range(int(n_samples/batch_size)):
        batch_x = X_train_m[batch*batch_size : (1+batch)*batch_size]
        batch_y = y_train_m[batch*batch_size : (1+batch)*batch_size]
        # print(batch_x)
        
        s.run([optimizer], feed_dict={input_X: batch_x, 
                                         input_y: batch_y,
                                         keep_prob:dropout})
        
        
    training_accuracy.append(s.run(accuracy, feed_dict= {input_X:X_train_m, 
                                                         input_y: y_train_m,
                                                         keep_prob:1
                                                         }))
    training_loss.append(s.run(loss, {input_X: X_train_m, 
                                      input_y: y_train_m,keep_prob:1}))
    
    # Evaluation of model
    testing_accuracy.append(accuracy_score(y_test_m.argmax(1), 
                            s.run(predicted_y, {input_X: X_test_m,keep_prob:1}).argmax(1)))
    print("Epoch:{0}, Train loss: {1:.2f} Train acc: {2:.5f}, Test acc:{3:.5f}".format(epoch, training_loss[epoch], training_accuracy[epoch],testing_accuracy[epoch]))
