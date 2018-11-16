import argparse
import datetime
import getpass
import logging
import os
import shutil
import threading
import time
from urllib import parse

import tensorflow as tf
import tensorflow.keras.backend as ktf
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.training import TrainSpec, EvalSpec, train_and_evaluate

import cifar10_data
import cifar10_model_cnn
import cifar10_model_resnet
from utils import ExamplesPerSecondHook

print(tf.__version__)
tf.logging.set_verbosity(tf.logging.DEBUG)
# start = time.time()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
ktf.set_session(sess)  # set this TensorFlow session as the default session for Keras

shapes = (32, 32, 3), 10
input_name = 'conv2d_input'

# tf.enable_eager_execution()

model_dir_hdfs = False
is_training = False
log = logging.getLogger('tensorflow')


def main(mname, model_dir, batch_size, epochs, eval_steps, eps_log_steps):
    global model_dir_hdfs
    if model_dir.startswith('hdfs'):
        model_dir_hdfs = True

    tf.logging.set_verbosity(tf.logging.DEBUG)
    # get TF logger
    log.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # create file handler which logs even debug messages
    if model_dir_hdfs is False:
        if os.path.exists(model_dir) is False:
            os.makedirs(model_dir)
        log_dir = model_dir
    else:
        model_dir = os.path.join(model_dir, "job_cifar10_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        log_dir = '.'

    # clear old log files
    with open(log_dir + '/tensorflow.log', 'w'):
        pass
    with open(log_dir + '/gpu.csv', 'w'):
        pass
    with open(log_dir + '/cpu.csv', 'w'):
        pass

    fh = logging.FileHandler(log_dir + '/tensorflow.log')

    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    log.info("TF version: %s", tf.__version__)
    log.info("Model directory: %s", model_dir)
    log.info("Batch size: %s", batch_size)
    log.info("Prefetch data all to memory: %s", True)
    log.info("Train epochs: %s", epochs)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    ktf.set_session(sess)  # set this TensorFlow session as the default session for Keras

    steps_per_epoch = cifar10_data.train_len() / batch_size
    log.info("Steps per epoch: %s", steps_per_epoch)
    if eval_steps is None:
        eval_steps = steps_per_epoch
    log.info("Evaluating each %i steps", eval_steps)

    if mname == "cnn":
        model = cifar10_model_cnn.cifar_model()
    else:
        model = cifar10_model_resnet.cifar_model()
        global input_name
        input_name = 'input_1'

    model.summary()

    def train_input_fn():
        dataset = tf.data.Dataset.from_generator(generator=cifar10_data.generator_train,
                                                 output_types=(tf.float32, tf.float32),
                                                 output_shapes=shapes)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=batch_size)
        # dataset = dataset.repeat(20)
        iterator = dataset.make_one_shot_iterator()
        features_tensors, labels = iterator.get_next()
        features = {input_name: features_tensors}
        return features, labels

    def eval_input_fn():
        dataset = tf.data.Dataset.from_generator(generator=cifar10_data.generator_test,
                                                 output_types=(tf.float32, tf.float32),
                                                 output_shapes=shapes)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=batch_size)
        iterator = dataset.make_one_shot_iterator()
        features_tensors, labels = iterator.get_next()
        features = {input_name: features_tensors}
        return features, labels

    my_config = RunConfig(
        save_checkpoints_steps=eval_steps  # Save checkpoints every n steps and run the evaluation.
        # keep_checkpoint_max = 5    # Retain the n most recent checkpoints (default 5).
    )
    estimator = tf.keras.estimator.model_to_estimator(model, config=my_config, model_dir=model_dir)

    examples_sec_hook = ExamplesPerSecondHook(batch_size, every_n_steps=eps_log_steps)
    # stopping_hook = early_stopping.stop_if_higher_hook(estimator, "accuracy", 0.5)

    train_hooks = [examples_sec_hook]

    train_spec = TrainSpec(input_fn=train_input_fn, hooks=train_hooks,
                           max_steps=cifar10_data.train_len() / batch_size * epochs)
    eval_spec = EvalSpec(input_fn=eval_input_fn, steps=cifar10_data.val_len() / batch_size,
                         throttle_secs=5)  # default 100 steps

    global is_training
    is_training = True
    threading.Thread(target=lambda: collect_stats(log_dir)).start()
    start = time.time()

    train_and_evaluate(estimator, train_spec, eval_spec)

    elapsed = time.time() - start
    is_training = False
    log.info("total time taken (seconds): %s ", elapsed)
    if model_dir_hdfs:
        parse_res = parse.urlsplit(model_dir)
        netloc = parse_res[1]
        path = parse_res[2]
        webhdfs_model_dir = 'http://' + netloc + ':50070/webhdfs/v1' + path
        username = getpass.getuser()
        component_name = estimator.config.task_type + str(estimator.config.task_id)
        log.info("Uploading log files for %s as %s to HDFS path: %s", component_name, username, webhdfs_model_dir)
        logging.shutdown()
        os.system('curl -L -i -T tensorflow.log "' + webhdfs_model_dir +
                  '/tensorflow-' + component_name + '.log?op=CREATE&overwrite=false&user.name=' + username + '"')
        os.system('curl -L -i -T cpu.csv "' + webhdfs_model_dir +
                  '/cpu-' + component_name + '.csv?op=CREATE&overwrite=false&user.name=' + username + '"')
        os.system('curl -L -i -T gpu.csv "' + webhdfs_model_dir +
                  '/gpu-' + component_name + '.csv?op=CREATE&overwrite=false&user.name=' + username + '"')
    else:
        log.info("Creating zip archive of job results")
        logging.shutdown()
        shutil.make_archive(model_dir, 'zip', model_dir)


def collect_stats(log_dir):
    log.info("Starting statistic collector")
    gpu_cmd = "echo $(date '+%Y-%m-%d %H:%M:%S'), $(nvidia-smi --format=csv,noheader " \
              "--query-gpu=power.draw,utilization.gpu,temperature.gpu) >> " + log_dir + "/gpu.csv"
    cpu_cmd = "echo $(date '+%Y-%m-%d %H:%M:%S'), $(ps -p " + str(os.getpid()) + \
              " -o %cpu,%mem --noheaders) >> " + log_dir + "/cpu.csv"
    while is_training:
        os.system(gpu_cmd)
        os.system(cpu_cmd)
        time.sleep(2)
    log.info("Finishing statistic collector")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mname',
        type=str,
        default="cnn",
        help='Model to use, cnn or resnet.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size.')
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs to train.')
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=None,
        help='Run the evaluation every n steps.')
    parser.add_argument(
        '--eps_log_steps',
        type=int,
        default=50,
        help='Log examples per second every n steps.')
    parser.add_argument(
        '--model_dir',
        type=str,
        default="job_cirfar10_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'),
        # default = "hdfs://l01abdpmn008.cnbdp.bmwgroup.net/user/amila/cifar10/model_dir"
        help='The directory where the checkpoint and summaries are stored.')
    args = parser.parse_args()

main(**vars(args))
