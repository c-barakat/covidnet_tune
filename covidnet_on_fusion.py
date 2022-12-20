from __future__ import print_function

import argparse
import os
import pathlib
import simplejson
import sys
import time
from functools import partial

sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from scipy.stats import reciprocal
import ray
import tensorflow as tf
from data import BalanceCovidDataset
from eval import eval
from ray import tune

from ray.tune.schedulers import (
    AsyncHyperBandScheduler,
    HyperBandScheduler,
    PopulationBasedTraining,
)


class TestLogger(tune.logger.Logger):
    def on_result(self, result):
        print("TestLogger", result)


def trial_str_creator(trial):
    return "{}_{}_123".format(trial.trainable_name, trial.trial_id)


class TrainableCovidNet(tune.Trainable):
    def _setup(self, config):
        learning_rate = config["learning_rate"]
        batch_size = config["batch_size"]
        class_weight_1 = config["class_weight_1"]
        class_weight_2 = config["class_weight_2"]
        class_weight_3 = config["class_weight_3"]
        covid_percent = config["covid_percent"]

        class_weights = [class_weight_1, class_weight_2, class_weight_3]
        weightspath = "covidnet_output/COVIDNet-CXR-Large"
        metaname = "model.meta"
        ckptname = "model-8485"
        n_classes = 3
        trainfile = "train_fusion_0.8.txt"
        testfile = "test_fusion_0.8.txt"
        self.input_size = 224
        self.out_tensorname = "dense_3/Softmax:0"
        self.in_tensorname = "input_1:0"
        logit_tensorname = "dense_3/MatMul:0"
        label_tensorname = "dense_3_target:0"
        weights_tensorname = "dense_3_sample_weights:0"

        self.display_step = 1
        outputPath = "covidnet_output/"
        self.runID = (
            "Fusion_COVIDNet-lr"
            + "{:.9f}".format(learning_rate)
            + "bs"
            + "{:.9f}".format(batch_size)
            + "cw[{}".format(class_weight_1)
            + ",{:9f}".format(class_weight_2)
            + ",{:9f}".format(class_weight_3)
            + "]"
        )
        self.runpath = outputPath + self.runID
        pathlib.Path(self.runpath).mkdir(parents=True, exist_ok=True)
        print("Output: " + self.runpath)

        with open(trainfile) as f:
            trainfiles = f.readlines()
        with open(testfile) as f:
            self.testfiles = f.readlines()

        self.mapping = {"normal": 0, "pneumonia": 1, "COVID-19": 2}

        self.generator = BalanceCovidDataset(
            data_dir="./data",
            csv_file=trainfile,
            batch_size=batch_size,
            input_shape=(self.input_size, self.input_size),
            n_classes=3,
            mapping=self.mapping,
            covid_percent=covid_percent,
            class_weights=class_weights,
            top_percent=0.08,
            is_severity_model=None,
        )

        tf.reset_default_graph()
        self.sess = tf.Session()

        self.saver = tf.train.import_meta_graph(os.path.join(weightspath, metaname))

        tf.get_default_graph()
        self.graph = tf.get_default_graph()

        self.image_tensor = self.graph.get_tensor_by_name(self.in_tensorname)
        self.labels_tensor = self.graph.get_tensor_by_name(label_tensorname)
        self.sample_weights = self.graph.get_tensor_by_name(weights_tensorname)
        self.pred_tensor = self.graph.get_tensor_by_name(logit_tensorname)

        # Define loss, accuracy and optimizer
        with tf.name_scope('loss'):
            loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.labels_tensor,
                logits=self.pred_tensor,
            )
        self.loss_op = tf.reduce_mean(loss_op*self.sample_weights)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate
            )
        self.train_op = optimizer.minimize(self.loss_op)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(
                tf.argmax(self.pred_tensor, 1),
                tf.argmax(self.labels_tensor, 1),
            )
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction)

        # Initialize the variables
        init = tf.global_variables_initializer()

        # Run the initializer
        self.sess.run(init)
        self.iterations = 0

        # save base model
        self.saver.save(self.sess, os.path.join(self.runpath, 'model'))

        print("Saved baseline checkpoint")
        print("Baseline eval:")
        eval(
            self.sess,
            self.graph,
            self.testfiles,
            os.path.join("./data", "test"),
            self.in_tensorname,
            self.out_tensorname,
            self.input_size,
            self.mapping,
        )

    def _train(self):

        total_batch = len(self.generator)
        progbar = tf.keras.utils.Progbar(total_batch)
        for i in range(total_batch):
            # Run optimization
            batch_x, batch_y, weights = next(self.generator)
            self.sess.run(
                self.train_op,
                feed_dict={
                    self.image_tensor: batch_x,
                    self.labels_tensor: batch_y,
                    self.sample_weights: weights,
                },
            )
            progbar.update(i + 1)

        pred = self.sess.run(
            self.pred_tensor, feed_dict={self.image_tensor: batch_x}
        )
        loss = self.sess.run(
            self.loss_op,
            feed_dict={
                self.pred_tensor: pred,
                self.labels_tensor: batch_y,
                self.sample_weights: weights,
            },
        )
        accuracy = self.sess.run(
            self.accuracy,
            feed_dict={
                self.pred_tensor: pred,
                self.labels_tensor: batch_y,
                self.sample_weights: weights,
            },
        )
        print(
            "Epoch:",
            "%04d" % (self.iterations + 1),
            "Minibatch loss=",
            "{:.9f}".format(loss),
            "Accuracy=",
            "{:.9f}".format(accuracy),
        )
        eval(
            self.sess,
            self.graph,
            self.testfiles,
            os.path.join("./data", "test"),
            self.in_tensorname,
            self.out_tensorname,
            self.input_size,
            self.mapping,
        )

        self.iterations += 1

        self.saver.save(
            self.sess,
            os.path.join(self.runpath, 'model'),
            global_step=self.iterations,
            write_meta_graph=False,
        )

        return {
            "Losses": loss,
            "Accuracies": accuracy,
            "Config": self.runID
        }

    def _save(self, checkpoint_dir):
        return self.saver.save(
            self.sess, checkpoint_dir + "/save", global_step=self.iterations
        )

    def _restore(self, path):
        return self.saver.restore(self.sess, path)

    def _stop(self):
        self.sess.close()


def get_scheduler(scheduler_name, num_epochs):
    if scheduler_name == 'pbt':
        return PopulationBasedTraining(
            time_attr="training_iteration",
            reward_attr="Losses",
            perturbation_interval=int(num_epochs / 2),
            hyperparam_mutations={
                "learning_rate": lambda lr: reciprocal.rvs(1e-5, 1e-3)
            },
        )
    if scheduler_name == 'ahb':
        return AsyncHyperBandScheduler(
            time_attr="training_iteration",
            reward_attr="Losses",
            grace_period=int(num_epochs / 2),
            max_t=num_epochs,
        )
    if scheduler_name == 'hpb':
        return HyperBandScheduler(
            time_attr="training_iteration",
            reward_attr="Losses",
            max_t=num_epochs,
        )


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1, cpus_per_trial=1, scheduler=None):
    ray.init(redis_address=os.environ['redis_total_address'])

    tune.register_trainable(f"{scheduler}_Train_COVID_Net", TrainableCovidNet)

    train_spec = {
        "run": f"{scheduler}_Train_COVID_Net",
        # Specify the number of CPU cores and GPUs each trial requires
        "resources_per_trial": {"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        "stop": {
            "training_iteration": max_num_epochs
        },
        # All your hyperparameters (variable and static ones)
        "config": {
            "batch_size": 8,
            "class_weight_1": 1,
            "class_weight_2": lambda w2: np.random.uniform(1, 10),
            "class_weight_3": lambda w3: np.random.uniform(1, 10),
            "covid_percent": lambda cp: np.random.uniform(0.05, 0.5),
            "learning_rate": lambda lr: reciprocal.rvs(1e-5, 1e-3),
        },
        "local_dir": "./trials/",
        # Number of trials
        "num_samples": num_samples,
    }

    try:
        result = tune.run_experiments(
            {f"{scheduler}_Train_COVID_Net": train_spec},
            scheduler=get_scheduler(scheduler, max_num_epochs),
        )
    except ValueError:
        print("errored trial")

`    results = {}
    for item in result:
        results[item.experiment_tag] = item.last_result

    with open(f"{scheduler}.json", "w") as f:
        simplejson.dump(results, f, ignore_nan=True)

    with open(f"{scheduler}_full_result.json", "w") as f:
        simplejson.dump(result, f, ignore_nan=True)

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        default=1,
        type=int,
        help="Number of trials to run.",
    )
    parser.add_argument(
        "--max_num_epochs",
        default=10,
        type=int,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--gpus_per_trial",
        default=1,
        type=int,
        help="Number of GPUs to dedicate for each trial.",
    )
    parser.add_argument(
        "--cpus_per_trial",
        default=1,
        type=int,
        help="Number of CPUs to dedicate for each trial.",
    )
    parser.add_argument(
        "--scheduler",
        default=None,
        type=str,
        help="Scheduler to be used during tuning.",
    )

    args = parser.parse_args()
    # You can change the number of GPUs per trial here:
    main(
        num_samples=args.num_samples,
        max_num_epochs=args.max_num_epochs,
        gpus_per_trial=args.gpus_per_trial,
        cpus_per_trial=args.cpus_per_trial,
        scheduler=args.scheduler,
    )
