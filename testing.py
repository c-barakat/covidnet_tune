import os
import sys
import time

import numpy as np
import pandas as pd

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import horovod.tensorflow as hvd

sys.path.append(os.getcwd())
from data import process_image_file

hvd.init()

rank = hvd.rank()
size = hvd.size()

config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.rank())
config.gpu_options.allow_growth = True


def mapping(a):
    if a['true'] == 'normal':
        return 0
    elif a['true'] == 'pneumonia':
        return 1
    else:
        return 2

    
def inv_mapping(a):
    if a['pred'] == 0:
        return 'normal'
    elif a['pred'] == 1:
        return 'pneumonia'
    else:
        return 'COVID-19'


def predict(sess, image_tensor, pred_tensor, x):

    pred = sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)})

    return pred.argmax(axis=1)[0]

if rank==0:
    trial = "trial01"
elif rank==1:
    trial = "trial02"
elif rank==2:
    trial = "trial03"
else:
    trial = "trial04"

print(f"Rank {rank} started inference on trial {trial}.")

weightspath = f"covidnet_output/{trial}"
metaname = 'model-50.meta'
ckptname = 'model-50'
in_tensorname = 'input_1:0'
out_tensorname = 'dense_3/Softmax:0'
input_size = 224
split = 'test'

df = pd.read_csv(
    "test_ehl_edited_0.8.txt",
    sep=" ",
    names=["ID", "filename", "condition"])

predictions = pd.DataFrame(columns=["pred", "time"])

top_percent = 0.08

tf.reset_default_graph()

tf.get_default_graph()

graph = tf.get_default_graph()

saver = tf.train.import_meta_graph(os.path.join(weightspath, metaname))

hooks = [hvd.BroadcastGlobalVariablesHook(0)]

with tf.train.MonitoredTrainingSession(
    config=config,
    hooks=hooks,
) as sess:

    for i in df.index:
        path = df.iloc[i]["filename"]
        imagepath = "./data/{}/{}".format(
            split, path
        )
        saver.restore(sess, os.path.join(weightspath, ckptname))
        
        image_tensor = graph.get_tensor_by_name(in_tensorname)
        pred_tensor = graph.get_tensor_by_name(out_tensorname)

        x = process_image_file(imagepath, top_percent, input_size)
        x = x.astype('float32') / 255.0

        t = time.time()
        pred = predict(
            sess,
            image_tensor,
            pred_tensor,
            x,
        )
        t = time.time() - t

        predictions.loc[i, "pred"] = pred
        predictions.loc[i, "time"] = t

predictions["true"] = df.condition
predictions["true_mapped"] = predictions.apply(mapping, axis=1)

predictions.to_csv(f"results/{trial}_predictions.csv", index=False)
print(f"Rank {rank} inference completed.")

barrier = hvd.allreduce(tf.random_normal(shape=[1]))
