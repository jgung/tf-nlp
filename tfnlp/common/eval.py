import re
import subprocess
import tempfile

import tensorflow as tf
from tensorflow.python.training import session_run_hook
from tensorflow.python.training.session_run_hook import SessionRunArgs

from tfnlp.common.constants import LABEL_KEY, LENGTH_KEY, PREDICT_KEY


def conll_eval(gold_batches, predicted_batches, script_path):
    """
    Run the CoNLL-2003 evaluation script on provided predicted sequences.
    :param gold_batches: list of gold label sequences
    :param predicted_batches: list of predicted label sequences
    :param script_path: path to CoNLL-2003 eval script
    :return: tuple of (overall F-score, script_output)
    """
    with tempfile.NamedTemporaryFile(mode='wt') as temp:
        for gold_seq, predicted_seq in zip(gold_batches, predicted_batches):
            for label, prediction in zip(gold_seq, predicted_seq):
                temp.write("_ {} {}\n".format(label, prediction))
            temp.write("\n")  # sentence break
        temp.flush()
        temp.seek(0)
        result = subprocess.check_output(["perl", script_path], stdin=temp, universal_newlines=True)
        return float(re.split('\s+', re.split('\n', result)[1].strip())[7]), result


class SequenceEvalHook(session_run_hook.SessionRunHook):
    def __init__(self, script_path, predict_tensor, gold_tensor, length_tensor, vocab):
        """
        Initialize a `SessionRunHook` used to perform off-graph evaluation of sequential predictions.
        :param script_path: path to eval script
        :param predict_tensor: iterable over a batch of predictions
        :param gold_tensor: iterable over the corresponding batch of labels
        :param length_tensor: batch-sized tensor of sequence lengths
        :param vocab: label feature vocab
        """
        self._script_path = script_path
        self._predict_tensor = predict_tensor
        self._gold_tensor = gold_tensor
        self._length_tensor = length_tensor
        self._vocab = vocab

        self._predictions = None
        self._gold = None
        self._best = -1

    def begin(self):
        self._predictions = []
        self._gold = []

    def before_run(self, run_context):
        fetches = {LABEL_KEY: self._gold_tensor,
                   PREDICT_KEY: self._predict_tensor,
                   LENGTH_KEY: self._length_tensor}
        return SessionRunArgs(fetches=fetches)

    def after_run(self, run_context, run_values):
        for gold, predictions, seq_len in zip(run_values.results[LABEL_KEY],
                                              run_values.results[PREDICT_KEY],
                                              run_values.results[LENGTH_KEY]):
            self._gold.append([self._vocab.index_to_feat(val) for val in gold][:seq_len])
            self._predictions.append([self._vocab.index_to_feat(val) for val in predictions][:seq_len])

    def end(self, session):
        if self._best >= 0:
            tf.logging.info("Current best score: %f", self._best)
        score, result = conll_eval(self._gold, self._predictions, self._script_path)
        tf.logging.info(result)
        if score > self._best:
            self._best = score
