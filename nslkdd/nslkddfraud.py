import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split

directory = './'
FIELD_NAMES_CSV = directory + 'Field Names.csv'
FIELDS = pd.read_csv(FIELD_NAMES_CSV, names=["field", "type"])
HEADER = list(FIELDS['field'])
HEADER.extend(["attack_type", "strength"])

def load_data(csv_path):    
    raw_data = pd.read_csv(csv_path, names=HEADER)

    # create one hot encoding from categorical variables
    flag_one_hot = pd.get_dummies(list(raw_data['flag']))
    protocol_type_one_hot = pd.get_dummies(list(raw_data['protocol_type']))
    
    # create new dataframe after filtering out data columns with low entropy and adding in one hot encoding
    processed_data = pd.concat([ raw_data[['land', 'num_compromised']], 
                                flag_one_hot, 
                                protocol_type_one_hot, 
                                raw_data[['diff_srv_rate', 'duration', 'src_bytes', 'dst_host_rerror_rate', 'dst_bytes']] ], 
                                axis=1)
  
   

    # dictionary that maps to new attack type
    mapped_dic = {'back': 'DoS', 
      'land': 'DoS', 
      'neptune': 'DoS', 
      'pod': 'DoS', 
      'smurf': 'DoS', 
      'teardrop': 'DoS',
      'buffer_overflow': 'U2R',
      'loadmodule': 'U2R',
      'perl': 'U2R',
      'rootkit': 'U2R',
      'ftp_write': 'R2L',
      'guess_passwd': 'R2L',
      'imap': 'R2L',
      'multihop': 'R2L',
      'phf': 'R2L',
      'spy': 'R2L',
      'warezmaster': 'R2L',
      'warezclient': 'R2L',
      'ipsweep': 'Probe',
      'nmap': 'Probe',
      'portsweep': 'Probe',
      'ipsweep': 'Probe',
      'satan': 'Probe',
      'normal': 'Normal',
     }

    # there are some values in the dataset that the author did not provide a new category for
    unmapped_values = list(set(pd.unique(raw_data['attack_type'])) - set(mapped_dic.keys()))
    
    unmapped_dic = {}
    for val in unmapped_values:
        unmapped_dic[val] = 'Other'
        
    map_to_new_attack_type = {**mapped_dic, **unmapped_dic}
    
    targets = raw_data['attack_type']
    targets.replace(map_to_new_attack_type, inplace=True)
    
    
    return processed_data, targets


# load training data
X, y = load_data(directory + 'KDDTrain+.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

X_train = X_train.values
y_train = pd.get_dummies(list(y_train))
y_train = y_train.values

X_test = X_test.values
y_test = pd.get_dummies(list(y_test))

y_test = y_test.values


s = tf.InteractiveSession()

# Defining various initialization parameters for 19-12-12-5 MLP model
num_classes = y_train.shape[1]
num_features = X_train.shape[1]
num_output = y_train.shape[1]
num_layers_0 = 13
num_layers_1 = 12
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


learning_rate = 0.005
# Adam optimizer for finding the right weight

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,var_list=[weights['0'], weights['1'],
                                                                         biases['0'], biases['1']])

# Metrics definition
correct_prediction = tf.equal(tf.argmax(y_train,1), tf.argmax(predicted_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Training parameters
batch_size = 500
epochs = 100
dropout = 0.4

training_accuracy = []
training_loss = []
testing_accuracy = []

s.run(tf.global_variables_initializer())
for epoch in range(epochs):    
    arr = np.arange(X_train.shape[0])
    np.random.shuffle(arr)
    for index in range(0,X_train.shape[0],batch_size):
        s.run(optimizer, {input_X: X_train[arr[index:index+batch_size]], 
                          input_y: y_train[arr[index:index+batch_size]],
                          keep_prob:dropout
                          })
    training_accuracy.append(s.run(accuracy, feed_dict= {input_X:X_train, 
                                                         input_y: y_train,
                                                         keep_prob:1
                                                         }))
    training_loss.append(s.run(loss, {input_X: X_train, 
                                      input_y: y_train,keep_prob:1}))
    
    # Evaluation of model
    testing_accuracy.append(accuracy_score(y_test.argmax(1), 
                            s.run(predicted_y, {input_X: X_test,keep_prob:1}).argmax(1)))
    print("Epoch:{0}, Train loss: {1:.2f} Train acc: {2:.3f}, Test acc:{3:.3f}".format(epoch, training_loss[epoch], training_accuracy[epoch],testing_accuracy[epoch]))
    