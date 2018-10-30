import argparse

import tensorflow as tf
from tensorflow.contrib.estimator.python.estimator import early_stopping
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.training import TrainSpec, EvalSpec, train_and_evaluate

import cifar10_model_cnn
import cifar10_data
import cifar10_model_resnet
from utils import ExamplesPerSecondHook
import tensorflow.keras.backend as ktf

print(tf.__version__)
tf.logging.set_verbosity(tf.logging.DEBUG)
# start = time.time()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
ktf.set_session(sess)  # set this TensorFlow session as the default session for Keras

batch_size = 32
data_dir = "data"
shapes = (32, 32, 3), 10
input_name = 'conv2d_input'


# tf.enable_eager_execution()


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


def main(mname, model_dir):
    if mname == "cnn":
        model = cifar10_model_cnn.cifar_model()
    else:
        model = cifar10_model_resnet.cifar_model()
        global input_name
        input_name = 'input_1'

    model.summary()

    my_config = RunConfig(
        save_checkpoints_steps=1000  # Save checkpoints every n steps and run the evaluation.
        # keep_checkpoint_max = 5    # Retain the n most recent checkpoints (default 5).
    )
    estimator = tf.keras.estimator.model_to_estimator(model, config=my_config, model_dir=model_dir)

    examples_sec_hook = ExamplesPerSecondHook(batch_size, every_n_steps=500)
    stopping_hook = early_stopping.stop_if_higher_hook(estimator, "accuracy", 0.5)

    train_hooks = [stopping_hook, examples_sec_hook]

    train_spec = TrainSpec(input_fn=train_input_fn, hooks=train_hooks)
    eval_spec = EvalSpec(input_fn=eval_input_fn, throttle_secs=5)
    # steps=len(x_test) / batch_size)  # default 100 steps

    train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mname',
        type=str,
        default="cnn",
        help='Model to use, cnn or resnet.')
    parser.add_argument(
        '--model_dir',
        type=str,
        default="job",
        help='The directory where the checkpoint and summaries are stored.')
    args = parser.parse_args()

main(**vars(args))
