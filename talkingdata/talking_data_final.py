import pandas as pd
import numpy as np
import warnings
import os
import gc

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'


path = ''
dtypes = {
        'ip'            : 'int64',
        'app'           : 'int32',
        'device'        : 'int32',
        'os'            : 'int32',
        'channel'       : 'int32',
        'is_attributed' : 'int16',
        'click_id'      : 'int64'
        }

train_df = pd.read_csv(path+"train2.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
test_df = pd.read_csv(path+"test2.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

len_train = len(train_df)

train_df=train_df.append(test_df)
del test_df; gc.collect()


train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
train_df['wday']  = pd.to_datetime(train_df.click_time).dt.dayofweek.astype('uint8')


gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})

train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')

del gp; gc.collect()

gp = train_df[['ip','app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
train_df = train_df.merge(gp, on=['ip','app'], how='left')
del gp; gc.collect()

gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
del gp; gc.collect()


train_df['qty'] = train_df['qty'].astype('int32')
train_df['ip_app_count'] = train_df['ip_app_count'].astype('int32')
train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('int32')


from sklearn.preprocessing import LabelEncoder
train_df[['app','device','os', 'channel', 'hour', 'day', 'wday']].apply(LabelEncoder().fit_transform) 

test_df = train_df[len_train:]
train_df = train_df[:len_train]

from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(train_df, test_size = 0.1)


train_df.drop(['click_id', 'click_time','ip'],1,inplace=True)

y_test = test_df['is_attributed'].values
test_df.drop(['click_id', 'click_time','ip'],1,inplace=True)


num_train_entries = train_df.shape[0]
num_train_features = train_df.shape[1] - 1
num_test_entries = test_df.shape[0]
num_test_features = test_df.shape[1] - 1


batch_size = 128

from tensorflow import feature_column
import numpy as np
import tensorflow as tf


n_emb = 50


max_app = np.max([train_df['app'].max(), test_df['app'].max()])+1
max_ch = np.max([train_df['channel'].max(), test_df['channel'].max()])+1
max_dev = np.max([train_df['device'].max(), test_df['device'].max()])+1
max_os = np.max([train_df['os'].max(), test_df['os'].max()])+1
max_h = np.max([train_df['hour'].max(), test_df['hour'].max()])+1
max_d = np.max([train_df['day'].max(), test_df['day'].max()])+1
max_wd = np.max([train_df['wday'].max(), test_df['wday'].max()])+1
max_qty = np.max([train_df['qty'].max(), test_df['qty'].max()])+1
max_c1 = np.max([train_df['ip_app_count'].max(), test_df['ip_app_count'].max()])+1
max_c2 = np.max([train_df['ip_app_os_count'].max(), test_df['ip_app_os_count'].max()])+1



app = feature_column.categorical_column_with_identity("app", max_app)
app_embedding = feature_column.embedding_column(app, dimension=n_emb)

channel = feature_column.categorical_column_with_identity("channel", max_ch)
channel_embedding = feature_column.embedding_column(channel, dimension=n_emb)

device = feature_column.categorical_column_with_identity("device", max_dev)
device_embedding = feature_column.embedding_column(device, dimension=n_emb)

os = feature_column.categorical_column_with_identity("os", max_os)
os_embedding = feature_column.embedding_column(os, dimension=n_emb)

hour = feature_column.categorical_column_with_identity("hour", max_h)
hour_embedding = feature_column.embedding_column(hour, dimension=n_emb)

day = feature_column.categorical_column_with_identity("day", max_d)
day_embedding = feature_column.embedding_column(day, dimension=n_emb)

wday = feature_column.categorical_column_with_identity("wday", max_wd)
wday_embedding = feature_column.embedding_column(wday, dimension=n_emb)

qty = feature_column.categorical_column_with_identity("qty", max_qty)
qty_embedding = feature_column.embedding_column(qty, dimension=n_emb)

c1 = feature_column.categorical_column_with_identity("ip_app_count", max_c1)
c1_embedding = feature_column.embedding_column(c1, dimension=n_emb)

c2 = feature_column.categorical_column_with_identity("ip_app_os_count", max_c2)
c2_embedding = feature_column.embedding_column(c2, dimension=n_emb)


feature_cols = [channel_embedding, device_embedding, os_embedding, hour_embedding, day_embedding, wday_embedding, qty_embedding, c1_embedding, c2_embedding]

dense_n = 1000   

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.estimator.DNNClassifier(
       feature_columns=feature_cols,
       hidden_units=[dense_n, dense_n],
       n_classes=2,
       dropout=0.6,
       optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
       model_dir="/tmp/talkingdata_model")

train_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=train_df,
        y=train_df['is_attributed'],
        batch_size=batch_size,
        num_epochs=100, # this way you can leave out steps from training
        shuffle= True
    )

classifier.train(input_fn=train_input_fn, steps=100)

val_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=val_df,
        y=val_df['is_attributed'],
        batch_size=batch_size,
        num_epochs=100,
        shuffle= True
    )

accuracy_score = classifier.evaluate(input_fn=val_input_fn, steps=25)

print(accuracy_score["accuracy"])