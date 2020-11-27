import pandas as pd
import tensorflow as tf

import numpy as np
import gc
from sklearn.metrics import roc_auc_score, accuracy_score


X_train = pd.read_csv("talkingdata-adtracking-fraud-detection/mnt/ssd/kaggle-talkingdata2/competition_files/train.csv", 
                      nrows=1000000, parse_dates=['click_time'])

X_train['day'] = X_train['click_time'].dt.day.astype('uint8')
X_train['hour'] = X_train['click_time'].dt.hour.astype('uint8')
X_train['minute'] = X_train['click_time'].dt.minute.astype('uint8')
X_train['second'] = X_train['click_time'].dt.second.astype('uint8')

ATTRIBUTION_CATEGORIES = [        
    # V1 Features #
    ###############
    ['ip'], ['app'], ['device'], ['os'], ['channel'],
    
    # V2 Features #
    ###############
    ['app', 'channel'],
    ['app', 'os'],
    ['app', 'device'],
    
    # V3 Features #
    ###############
    ['channel', 'os'],
    ['channel', 'device'],
    ['os', 'device']
]

freqs = {}
for cols in ATTRIBUTION_CATEGORIES:
    
    # New feature name
    new_feature = '_'.join(cols)+'_confRate'    
    
    # Perform the groupby
    group_object = X_train.groupby(cols)
    
    # Group sizes    
    group_sizes = group_object.size()
    
    log_group = np.log(100000) # 1000 views -> 60% confidence, 100 views -> 40% confidence 
    print(">> Calculating confidence-weighted rate for: {}.\n   Saving to: {}. Group Max /Mean / Median / Min: {} / {} / {} / {}".format(
        cols, new_feature, 
        group_sizes.max(), 
        np.round(group_sizes.mean(), 2),
        np.round(group_sizes.median(), 2),
        group_sizes.min()
    ))
    
    # Aggregation function
    def rate_calculation(x):
        """Calculate the attributed rate. Scale by confidence"""
        rate = x.sum() / float(x.count())
        conf = np.min([1, np.log(x.count()) / log_group])
        return rate * conf
    
    # Perform the merge
    X_train = X_train.merge(
        group_object['is_attributed']. \
            apply(rate_calculation). \
            reset_index(). \
            rename( 
                index=str,
                columns={'is_attributed': new_feature}
            )[cols + [new_feature]],
        on=cols, how='left'
    )

X_labels = X_train['is_attributed']

X_labels = pd.get_dummies(list(X_labels))

X_train_processed = X_train.drop(columns = ['is_attributed', 'click_time', 'attributed_time'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train_processed, X_labels, test_size = 0.1)

X_train = X_train.values
y_train = y_train.values

X_test = X_test.values
y_test = y_test.values


s = tf.InteractiveSession()
num_classes = y_train.shape[1]
num_features = X_train.shape[1]


num_output = y_train.shape[1]
num_layers_0 = 13
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
    print("Epoch:{0}, Train loss: {1:.2f} Train acc: {2:.5f}, Test acc:{3:.5f}".format(epoch, training_loss[epoch], training_accuracy[epoch],testing_accuracy[epoch]))