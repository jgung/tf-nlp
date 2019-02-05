import argparse
import os
import sys

import tensorflow as tf
from tensorflow.contrib.estimator import stop_if_no_increase_hook
from tensorflow.contrib.predictor import from_saved_model
from tensorflow.contrib.training import HParams
from tensorflow.python.estimator.export.export import ServingInputReceiver
from tensorflow.python.estimator.exporter import BestExporter
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.training import train_and_evaluate
from tensorflow.python.framework import dtypes
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops

from tfnlp.cli.evaluators import get_evaluator
from tfnlp.common import constants
from tfnlp.common.config import get_network_config
from tfnlp.common.eval import metric_compare_fn
from tfnlp.common.logging import set_up_logging
from tfnlp.common.utils import read_json, write_json
from tfnlp.datasets import make_dataset, padded_batch
from tfnlp.feature import get_default_buckets, get_feature_extractor, write_features
from tfnlp.model.model import multi_head_model_func
from tfnlp.predictor import get_latest_savedmodel_from_jobdir, from_job_dir
from tfnlp.readers import get_reader


def default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', dest='save', type=str, required=True,
                        help='models/checkpoints/vocabularies save path')
    parser.add_argument('--config', type=str, help='training configuration JSON')
    parser.add_argument('--resources', type=str, help='shared resources directory (such as for word embeddings)')
    parser.add_argument('--train', type=str, help='training data path')
    parser.add_argument('--valid', type=str, help='validation/development data path')
    parser.add_argument('--test', type=str, help='test data paths, comma-separated')
    parser.add_argument('--mode', type=str, default="train", help='(optional) training command, "train" by default',
                        choices=['train', 'test', 'predict', 'itl'])
    parser.add_argument('--script', type=str, help='(optional) evaluation script path')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='overwrite previously saved vocabularies and training files')
    parser.add_argument('--output', type=str, help='output path for predictions (during evaluation and application)')
    parser.set_defaults(overwrite=False)
    return parser


class Trainer(object):
    def __init__(self, args=None):
        super().__init__()
        args = self._validate_and_parse_args(args)
        self._job_dir = args.save
        self._mode = args.mode
        if self._mode == 'train' and not args.train and args.test:
            self._mode = 'test'
        self._raw_train = args.train
        self._raw_valid = args.valid
        self._raw_test = [t for t in args.test.split(',') if t.strip()] if args.test else None
        self._overwrite = args.overwrite
        if args.output:
            self._output = os.path.join(args.save, args.output)
        else:
            self._output = os.path.join(args.save, 'predictions.txt')

        self._save_path = os.path.join(args.save, constants.MODEL_PATH)
        self._vocab_path = os.path.join(args.save, constants.VOCAB_PATH)
        self._resources = args.resources
        self._eval_script_path = args.script

        # read configuration file
        self.config_path = os.path.join(args.save, constants.CONFIG_PATH)
        if not tf.gfile.Exists(self.config_path) or self._overwrite:
            if not args.config:
                raise AssertionError('"--config" option is required when training for the first time')
            tf.gfile.MakeDirs(args.save)
            tf.gfile.Copy(args.config, self.config_path, overwrite=True)
        self._training_config = get_network_config(read_json(self.config_path))
        self._feature_config = self._training_config.features

        self._model_fn = get_model_func(self._training_config)

        self._feature_extractor = None
        self._estimator = None

        self._raw_instance_reader_fn = lambda raw_path: get_reader(self._training_config.reader,
                                                                   self._training_config).read_file(raw_path)
        # TODO: use separate test config
        self._raw_test_instance_reader_fn = lambda raw_path: get_reader(self._training_config.reader).read_file(raw_path)
        self._data_path_fn = lambda orig: os.path.join(args.save, os.path.basename(orig) + ".tfrecords")

        set_up_logging(os.path.join(args.save, '{}.log'.format(self._mode)))

    # noinspection PyMethodMayBeStatic
    def _validate_and_parse_args(self, args):
        """
        Parse arguments using argument parser. Can override for additional validation or logic.
        :param args: command-line arguments
        :return: parsed arguments
        """
        parser = default_args()
        if len(sys.argv) == 1:
            parser.print_help(sys.stderr)
            sys.exit(1)
        return parser.parse_args(args)

    def run(self):
        self._init_feature_extractor()
        if self._raw_train and self._mode == "train":
            self.train()
        elif self._mode == "predict":
            self.predict()
        elif self._mode == "itl":
            self.itl()
        elif self._mode == "test":
            self.eval()
        else:
            raise ValueError("Unexpected mode type: {}".format(self._mode))

    def train(self):
        self._init_estimator(test=False)

        # TODO: fixes issue https://github.com/tensorflow/tensorflow/issues/18394
        if not os.path.exists(self._estimator.eval_dir()):
            os.makedirs(self._estimator.eval_dir())

        early_stopping = stop_if_no_increase_hook(
            self._estimator,
            metric_name=self._training_config.metric,
            max_steps_without_increase=self._training_config.patience,
            min_steps=100,
            run_every_secs=None,
            run_every_steps=100,
        )

        exporter = BestExporter(serving_input_receiver_fn=self._serving_input_fn,
                                compare_fn=metric_compare_fn(self._training_config.metric),
                                exports_to_keep=self._training_config.exports_to_keep)

        tf.logging.info('Training on %s, validating on %s' % (self._raw_train, self._raw_valid))
        self._extract_and_write(self._raw_train)
        self._extract_and_write(self._raw_valid)

        train_input_fn = self._input_fn(self._raw_train, True)
        valid_input_fn = self._input_fn(self._raw_valid, False)

        train_and_evaluate(self._estimator,
                           train_spec=tf.estimator.TrainSpec(input_fn=train_input_fn,
                                                             max_steps=self._training_config.max_steps,
                                                             hooks=[early_stopping]),
                           eval_spec=tf.estimator.EvalSpec(input_fn=valid_input_fn,
                                                           steps=None,
                                                           exporters=[exporter],
                                                           throttle_secs=0))
        if self._raw_test:
            self.eval()

    def eval(self):
        predictor = from_job_dir(self._job_dir)
        evaluator = get_evaluator(self._training_config)
        for test_set in self._raw_test:
            tf.logging.info('Evaluating on %s' % test_set)
            instances = list(self._extract_raw(test_set, True))
            processed_examples = predictor.predict_inputs(instances, formatted=False)
            evaluator(instances, processed_examples, os.path.join(self._save_path, os.path.basename(test_set) + '.eval'))

    def predict(self):
        predictor = from_job_dir(self._job_dir)
        for test_set in self._raw_test:
            prediction_path = test_set + '.predictions.txt'
            tf.logging.info('Writing predictions on %s to %s' % (test_set, prediction_path))
            with file_io.FileIO(prediction_path, mode="w") as output:
                with file_io.FileIO(test_set, mode="r") as text_lines:
                    for line in text_lines:
                        line = line.strip()
                        if not line:
                            continue
                        predictions = predictor.predict(line)
                        for prediction in predictions:
                            output.write(str(prediction) + '\n')
                        output.write('\n')

    def itl(self):
        predictor = from_job_dir(self._job_dir)
        while True:
            sentence = input(">>> ")
            if not sentence:
                continue
            if sentence.lower() in {'exit', 'quit'}:
                break
            predictions = predictor.predict(sentence)
            for prediction in predictions:
                print(str(prediction))

    def _get_predictor(self):
        latest = get_latest_savedmodel_from_jobdir(self._job_dir)
        tf.logging.info("Loading predictor from saved model at %s" % latest)
        return from_saved_model(latest)

    def _init_feature_extractor(self):
        self._feature_extractor = get_feature_extractor(self._feature_config)
        if self._mode == "train":
            if not self._overwrite:
                tf.logging.info("Checking for pre-existing vocabulary at vocabulary at %s", self._vocab_path)
                if self._feature_extractor.read_vocab(self._vocab_path):
                    tf.logging.info("Loaded pre-existing vocabulary at %s", self._vocab_path)
                else:
                    tf.logging.info("No valid pre-existing vocabulary found at %s "
                                    "(this is normal when not loading from an existing model)", self._vocab_path)
                    self._train_vocab()
            else:
                self._train_vocab()
        else:
            tf.logging.info("Checking for pre-existing vocabulary at vocabulary at %s", self._vocab_path)
            self._feature_extractor.read_vocab(self._vocab_path)
            tf.logging.info("Loaded pre-existing vocabulary at %s", self._vocab_path)

    def _extract_raw(self, path, test=False):
        if test:
            raw_instances = self._raw_test_instance_reader_fn(path)
        else:
            raw_instances = self._raw_instance_reader_fn(path)
        if not raw_instances:
            raise ValueError("No examples provided at path given by '{}'".format(path))
        return raw_instances

    def _train_vocab(self):
        tf.logging.info("Training new vocabulary using training data at %s", self._raw_train)
        self._feature_extractor.initialize(self._resources)
        self._feature_extractor.train(self._extract_raw(self._raw_train))
        self._feature_extractor.write_vocab(self._vocab_path, overwrite=self._overwrite, resources=self._resources, prune=True)

    def _extract_features(self, path, test=False):
        tf.logging.info("Extracting features from %s", path)
        examples = self._feature_extractor.extract_all(self._extract_raw(path, test))
        return examples

    def _extract_and_write(self, path, test=False):
        output_path = self._data_path_fn(path)
        if tf.gfile.Exists(output_path) and not self._overwrite:
            tf.logging.info("Using existing features for %s from %s", path, output_path)
            return
        examples = self._extract_features(path, test)
        tf.logging.info("Writing extracted features from %s for %d instances to %s", path, len(examples), output_path)
        write_features(examples, output_path)

    def _init_estimator(self, test=False, tag=None):
        self._estimator = tf.estimator.Estimator(model_fn=self._model_fn, model_dir=self._save_path,
                                                 config=RunConfig(
                                                     keep_checkpoint_max=self._training_config.keep_checkpoints,
                                                     save_checkpoints_steps=self._training_config.checkpoint_steps),
                                                 params=self._params(test=test, tag=tag))

    def _serving_input_fn(self):
        # input has been serialized to a TFRecord string (variable batch size)
        serialized_tf_example = array_ops.placeholder(dtype=dtypes.string, shape=[None], name=constants.SERVING_PLACEHOLDER)
        # create single padded batch
        batch = padded_batch(self._feature_extractor, serialized_tf_example)
        return ServingInputReceiver(batch, {"examples": serialized_tf_example})

    def _params(self, test=False, tag=None):
        return HParams(extractor=self._feature_extractor,
                       config=self._training_config,
                       script_path=self._eval_script_path,
                       vocab_path=self._vocab_path,
                       output=self._output if not tag else self._output + '.' + tag,
                       verbose_eval=test)

    def _input_fn(self, dataset, train=False):
        bucket_sizes = self._training_config.bucket_sizes
        if not bucket_sizes and constants.LENGTH_KEY in self._feature_extractor.features:
            length_feat = self._feature_extractor.feature(constants.LENGTH_KEY)
            bucket_sizes = get_default_buckets(length_feat.counts, self._training_config.batch_size * 2,
                                               max_length=self._training_config.max_length)
            if not bucket_sizes:
                bucket_sizes = None
            else:
                # persist dynamically computed bucket sizes
                self._training_config.bucket_sizes = bucket_sizes
                write_json(self._training_config, self.config_path)

        return lambda: make_dataset(self._feature_extractor, paths=self._data_path_fn(dataset),
                                    batch_size=self._training_config.batch_size, evaluate=not train,
                                    bucket_sizes=bucket_sizes)


def get_model_func(config):
    head_type = [head.type for head in config.heads][0]
    model_funcs = {
        constants.CLASSIFIER_KEY: multi_head_model_func,
        constants.TAGGER_KEY: multi_head_model_func,
        constants.NER_KEY: multi_head_model_func,
        constants.PARSER_KEY: multi_head_model_func,
        constants.SRL_KEY: multi_head_model_func,
        constants.TOKEN_CLASSIFIER_KEY: multi_head_model_func,
    }
    if head_type not in model_funcs:
        raise ValueError("Unexpected head type: " + head_type)
    return model_funcs[head_type]


if __name__ == '__main__':
    Trainer().run()
