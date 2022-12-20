from __future__ import print_function
# from mpi4py import MPI
import tensorflow as tf
import os, argparse, pathlib
import sys
import time
import pandas as pd
import numpy as np
import horovod.tensorflow as hvd

sys.path.append(os.getcwd())

from eval import eval
from data import BalanceCovidDataset

# comm = MPI.COMM_WORLD
tf.logging.set_verbosity(tf.logging.ERROR)

hvd.init()
config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())


def covidnet_train(
    epochs,
    learning_rate,
    batch_size,
    weightspath,
    metaname,
    ckptname,
    trainfile,
    testfile,
    n_classes,
    class_weights,
    covid_percent,
    input_size,
    out_tensorname,
    in_tensorname,
    logit_tensorname,
    label_tensorname,
    weights_tensorname,
    mapping,
):

    display_step = 1
    outputPath = './covidnet_output/'
    runID = 'COVIDNet-lr' + str(learning_rate) + "hvd1"
    runPath = outputPath + runID
    pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
    print('Output: ' + runPath)

#     with open(trainfile) as f:
#         trainfiles = f.readlines()
    with open(testfile) as f:
        testfiles = f.readlines()

#     mapping = {
#         'normal': 0,
#         'pneumonia': 1,
#         'COVID-19': 2
#     }

    generator = BalanceCovidDataset(
        data_dir="./data",
        csv_file=trainfile,
        batch_size=batch_size,
        input_shape=(input_size, input_size),
        n_classes=n_classes,
        mapping=mapping,
        covid_percent=covid_percent,
        class_weights=class_weights,
        top_percent=0.08,
        is_severity_model=None,
    )

    tf.reset_default_graph()
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(weightspath, metaname))

    graph = tf.get_default_graph()

    image_tensor = graph.get_tensor_by_name(in_tensorname)
    labels_tensor = graph.get_tensor_by_name(label_tensorname)
    sample_weights = graph.get_tensor_by_name(weights_tensorname)
    pred_tensor = graph.get_tensor_by_name(logit_tensorname)
    # loss expects unscaled logits since it performs a softmax on logits internally for efficiency

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=pred_tensor, labels=labels_tensor)*sample_weights)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # optimize through HVD
    optimizer = hvd.DistributedOptimizer(optimizer)
    train_op = optimizer.minimize(loss_op)

    hooks = [hvd.BroadcastGlobalVariablesHook(0)]

    with tf.train.MonitoredTrainingSession(
        config=config,
        hooks=hooks,
    ) as sess:
        # load weights
        saver.restore(sess, os.path.join(weightspath, ckptname))

        # Training cycle
        print('Training started')
        total_batch = len(generator)
        progbar = tf.keras.utils.Progbar(total_batch)

        losses = []

        for epoch in range(epochs):
            for i in range(total_batch):
                # Run optimization
                batch_x, batch_y, weights = next(generator)

                sess.run(train_op, feed_dict={image_tensor: batch_x,
                                              labels_tensor: batch_y,
                                              sample_weights: weights})
                progbar.update(i+1)

            if epoch % display_step == 0:
                pred = sess.run(pred_tensor, feed_dict={image_tensor:batch_x})
                loss = sess.run(loss_op, feed_dict={pred_tensor: pred,
                                                    labels_tensor: batch_y,
                                                    sample_weights: weights})
                print("Epoch:", '%04d' % (epoch + 1), "Minibatch loss=", "{:.9f}".format(loss))
                losses.append(loss)
                eval(
                    sess,
                    graph,
                    testfiles,
                    os.path.join("./data", "test"),
                    in_tensorname,
                    out_tensorname,
                    input_size,
                    mapping,
                )

    if hvd.rank() == 0:
        print("Optimization Finished!")
        print("Losses: {}".format(losses))
        return loss


def split_data(trainfile, size, rank, mapping):
    filename = f"trainset_size{size}_rank{rank}.txt"
    with open(trainfile, "r") as fr:
        traindata = fr.readlines()

    datasets = {}
    for key in mapping.keys():
        datasets[key] = []

    for l in traindata:
        datasets[l.split()[2]].append(l)

    datasets = [datasets["normal"] + datasets["pneumonia"], datasets["COVID-19"]]

    rank_data = [
        datasets[0][
            rank
            * int(len(datasets[0]) / size) : (rank + 1)
            * int(len(datasets[0]) / size)
        ],
        datasets[1][
            rank
            * int(len(datasets[1]) / size) : (rank + 1)
            * int(len(datasets[1]) / size)
        ],
    ]
    
    rank_data = rank_data[0] + rank_data[1]
    
    with open(filename, "w") as fw:
        fw.write("".join(rank_data))
        fw.close()

    return filename


def main(run_id, learning_rate, class_weights, covid_percent):
    data = np.empty([0, 2])

    epochs = 25
    learning_rate = learning_rate*(hvd.size())
    batch_size = 8
    weightspath = "covidnet_output/COVIDNet-CXR-Large"
    metaname = 'model.meta'
    ckptname = 'model-8485'
    n_classes = 3
    trainfile = "train_fusion_0.8.txt"
    testfile = "test_fusion_0.8.txt"
    class_weights = class_weights
    covid_percent = covid_percent
    input_size = 224
    out_tensorname = "dense_3/Softmax:0"
    in_tensorname = "input_1:0"
    logit_tensorname = "dense_3/MatMul:0"
    label_tensorname = "dense_3_target:0"
    weights_tensorname = "dense_3_sample_weights:0"

    mapping = {
        'normal': 0,
        'pneumonia': 1,
        'COVID-19': 2
    }

    training_data_file = split_data(trainfile, hvd.size(), hvd.rank(), mapping)

    t = time.time()
    loss = covidnet_train(
        epochs,
        learning_rate,
        batch_size,
        weightspath,
        metaname,
        ckptname,
        training_data_file,
        testfile,
        n_classes,
        class_weights,
        covid_percent,
        input_size,
        out_tensorname,
        in_tensorname,
        logit_tensorname,
        label_tensorname,
        weights_tensorname,
        mapping,
    )
    t = time.time() - t

    if hvd.rank() == 0:
        data = np.concatenate((data, np.array([[loss, t]])), axis=0)

        df_data = pd.DataFrame(data, columns=['loss', 'time'])
#         df_out = df_data.drop('index', axis=1)
        df_data.to_csv(
            f"training_run_size_{hvd.size()}.csv",
            index=False
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_id",
        default=1,
        type=int,
        help="Identifier of the trial.",
    )
    parser.add_argument(
        "--learning_rate",
        default=0.1,
        type=float,
        help="Learning rate for training.",
    )
    parser.add_argument(
        "--class_weights",
        default=[1, 1, 1],
        nargs='+',
        type=float,
        help="Weights for each class to be used.",
    )
    parser.add_argument(
        "--covid_percent",
        default=0.5,
        type=float,
        help="Covid-19 percentage.",
    )

    args = parser.parse_args()

    main(
        run_id=args.run_id,
        learning_rate=args.learning_rate,
        class_weights=args.class_weights,
        covid_percent=args.covid_percent,
    )
