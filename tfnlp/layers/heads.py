import os

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.lookup_ops import index_to_string_table_from_file
from tensorflow_addons.text import crf

from tfnlp.cli.evaluators import TaggerEvaluator, SrlEvaluator, TokenClassifierEvaluator
from tfnlp.common import constants
from tfnlp.common.bert import BERT_SUBLABEL
from tfnlp.common.config import append_label
from tfnlp.common.eval_hooks import ClassifierEvalHook, SequenceEvalHook, SrlEvalHook
from tfnlp.layers.layers import string2index, get_encoder_input
from tfnlp.layers.util import get_shape, mlp, bilinear, select_logits, sequence_loss


class ModelHead(object):
    def __init__(self, inputs, config, features, params, training):
        self.inputs = inputs
        self.config = config
        self.name = config.name
        self.extractor = params.extractor.targets[self.name]
        self.index = params.extractor.ordered_targets.index(self.name)
        self.features = features
        self.params = params
        self._training = training

        self.targets = None
        self.logits = None
        self.loss = None
        self.predictions = None
        self.evaluation_hooks = []
        self.metric_ops = {}
        self.metric = None
        self.export_outputs = {}

    def training(self):
        self.targets = self.features[self.name]
        if self.extractor.has_vocab():
            self.targets = string2index(self.features[self.name], self.extractor)

        with tf.compat.v1.variable_scope(self.name):
            self._all()
            self._train_eval()
            self._train()

    def evaluation(self):
        self.targets = self.features[self.name]
        if self.extractor.has_vocab():
            self.targets = string2index(self.features[self.name], self.extractor)

        with tf.compat.v1.variable_scope(self.name):
            self._all()
            self._train_eval()
            self._eval_predict()
            self._evaluation()

    def prediction(self):
        with tf.compat.v1.variable_scope(self.name):
            self._all()
            self._eval_predict()
            self._prediction()

    def _all(self):
        """
        Called for every setting (training/evaluation/prediction).
        """
        pass

    def _train_eval(self):
        """
        Called after `_all` for training/evaluation.
        """
        pass

    def _train(self):
        """
        Called after `_train_eval` for training.
        """
        pass

    def _eval_predict(self):
        """
        Called after `_train_eval` for evaluation and after `_all` for prediction.
        """
        pass

    def _evaluation(self):
        """
        Called after `_eval_predict` for evaluation.
        """
        pass

    def _prediction(self):
        """
        Called after `_eval_predict` for prediction.
        """
        index_to_label = index_to_string_table_from_file(vocabulary_file=os.path.join(self.params.vocab_path, self.name),
                                                         default_value=self.extractor.unknown_word)
        self.predictions = tf.identity(index_to_label.lookup(tf.cast(self.predictions, dtype=tf.int64)), name="labels")
        self.export_outputs = {self.name: self.predictions}


class ClassifierHead(ModelHead):
    def __init__(self, inputs, config, features, params, training):
        super().__init__(inputs, config, features, params, training)
        self._sequence_lengths = self.features[constants.LENGTH_KEY]
        self.scores = None

    def _all(self):
        if len(self.inputs) == 2:
            inputs = tf.squeeze(self.inputs[0], axis=1)
        else:
            inputs = self.inputs[2]
            inputs = tf.compat.v1.layers.dropout(inputs, training=self._training)

        with tf.compat.v1.variable_scope("logits"):
            num_labels = self.extractor.vocab_size()
            self.logits = tf.compat.v1.layers.dense(inputs, num_labels, kernel_initializer=tf.compat.v1.zeros_initializer)

    def _train_eval(self):
        self.loss = tf.cond(pred=tf.reduce_sum(input_tensor=self.features[constants.ACTIVE_TASK_KEY], axis=0)[self.index] > 0,
                            true_fn=self._loss,
                            false_fn=lambda: tf.constant(0, dtype=tf.float32))
        self.metric = tf.Variable(0, name=append_label(constants.ACCURACY_METRIC_KEY, self.name), dtype=tf.float32,
                                  trainable=False)

    def _loss(self):
        if self.config.label_smoothing > 0:
            targets = tf.one_hot(self.targets, depth=self.extractor.vocab_size())
            return tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=targets,
                                                             logits=self.logits,
                                                             label_smoothing=self.config.label_smoothing)
        else:
            return tf.reduce_mean(
                input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets))

    def _eval_predict(self):
        self.scores = tf.nn.softmax(self.logits)  # (b x n)
        self.predictions = tf.argmax(input=self.logits, axis=1)

    def training(self):
        super().training()
        self._eval_predict()
        self._prediction()

    def evaluation(self):
        super().evaluation()
        self._prediction()

    def _evaluation(self):
        predictions_key = append_label(constants.PREDICT_KEY, self.name)
        labels_key = append_label(constants.LABEL_KEY, self.name)
        acc_key = append_label(constants.ACCURACY_METRIC_KEY, self.name)

        self.metric_ops = {
            acc_key: tf.compat.v1.metrics.accuracy(labels=self.targets, predictions=self.predictions, name=acc_key)}

        tensors = {
            labels_key: self.targets,
            predictions_key: self.predictions,
            constants.LABEL_SCORES: self.scores,
            constants.LENGTH_KEY: self._sequence_lengths,
            constants.SENTENCE_INDEX: self.features[constants.SENTENCE_INDEX],
            constants.ACTIVE_TASK_KEY: self.features[constants.ACTIVE_TASK_KEY]
        }

        constraint_key = self.extractor.constraint_key
        if constraint_key:
            tensors[constraint_key] = self.features[constraint_key]

        overall_score = tf.identity(self.metric)
        self.metric_ops[append_label(constants.ACCURACY_METRIC_KEY, self.name)] = (overall_score, overall_score)
        overall_key = append_label(constants.ACCURACY_METRIC_KEY, self.name)
        # https://github.com/tensorflow/tensorflow/issues/20418 -- metrics don't accept variables, so we create a tensor
        eval_placeholder = tf.compat.v1.placeholder(dtype=tf.float32, name='update_%s' % overall_key)

        self.evaluation_hooks = [
            ClassifierEvalHook(
                index=self.index,
                label_key=labels_key,
                predict_key=predictions_key,
                tensors=tensors,
                evaluator=TokenClassifierEvaluator(
                    target=self.extractor,
                    output_path=os.path.join(self.params.job_dir, self.name + '.dev')),
                output_dir=self.params.job_dir,
                eval_update=tf.compat.v1.assign(self.metric, eval_placeholder),
                eval_placeholder=eval_placeholder,
            )
        ]

    def _prediction(self):
        super()._prediction()
        self.export_outputs[append_label(constants.LABEL_SCORES, self.name)] = self.scores


def select_by_token_index(states, indices):
    row_indices = tf.range(tf.shape(input=indices, out_type=tf.int64)[0])
    full_indices = tf.stack([row_indices, indices], axis=1)
    return tf.gather_nd(states, indices=full_indices)


class TokenClassifierHead(ClassifierHead):
    def __init__(self, inputs, config, features, params, training):
        super().__init__(inputs, config, features, params, training)

    def _all(self):
        inputs = get_encoder_input(self.inputs)

        if constants.TOKEN_INDEX_KEY in self.features:
            targets = self.features[constants.TOKEN_INDEX_KEY]
        else:
            targets = self.features[constants.PREDICATE_INDEX_KEY]
        inputs = select_by_token_index(inputs, targets)

        with tf.compat.v1.variable_scope("logits"):
            num_labels = self.extractor.vocab_size()
            self.logits = tf.compat.v1.layers.dense(inputs, num_labels, kernel_initializer=tf.compat.v1.zeros_initializer)


def create_transition_matrix(labels):
    """
    Return a numpy matrix to enforce valid transitions for IOB-style tagging problems.
    :param labels: label feature extractor
    """
    labels = [labels.index_to_feat(i) for i in range(len(labels.indices))]
    num_tags = len(labels)
    transition_params = np.zeros([num_tags, num_tags], dtype=np.float32)
    for i, prev_label in enumerate(labels):
        for j, curr_label in enumerate(labels):
            if curr_label == BERT_SUBLABEL:
                transition_params[i, j] = np.NINF
            elif i == j:
                continue
            elif curr_label[:2] == 'I-' and prev_label != 'B-' + curr_label[2:]:
                transition_params[i, j] = np.NINF
    return tf.compat.v1.initializers.constant(transition_params)


class TaggerHead(ModelHead):

    def __init__(self, inputs, config, features, params, training=False):
        super().__init__(inputs, config, features, params, training)
        if constants.BERT_SPLIT_INDEX in self.features and constants.BERT_LENGTH_KEY not in self.features:
            self._sequence_lengths = self.features[constants.BERT_SPLIT_INDEX]
        else:
            self._sequence_lengths = self.features[constants.LENGTH_KEY]

        self._tag_transitions = None

    def training(self):
        super().training()
        self._eval_predict()
        self._prediction()

    def evaluation(self):
        super().evaluation()
        self._prediction()

    def _all(self):
        inputs = get_encoder_input(self.inputs)
        input_shape = get_shape(inputs)  # (b x n x d), d == output_size
        time_steps, encoder_dim = input_shape[1], input_shape[2]

        # flatten encoder outputs to a (batch_size * time_steps x encoder_dim) Tensor for batch matrix multiplication
        inputs = tf.reshape(inputs, [-1, encoder_dim], name="flatten")

        with tf.compat.v1.variable_scope("logits"):
            num_labels = self.extractor.vocab_size()
            initializer = tf.compat.v1.zeros_initializer if self.config.zero_init else tf.compat.v1.random_normal_initializer(
                stddev=0.01)

            dense = tf.compat.v1.layers.dense(inputs, num_labels, kernel_initializer=initializer)
            # batch multiplication complete, convert back to a (batch_size x time_steps x num_labels) Tensor
            self.logits = tf.reshape(dense, [-1, time_steps, num_labels], name="unflatten")
        if self.config.crf:
            # explicitly train a transition matrix
            self._tag_transitions = tf.compat.v1.get_variable("transitions", [num_labels, num_labels])
        else:
            # use constrained decoding based on IOB labels
            self._tag_transitions = tf.compat.v1.get_variable("transitions", [num_labels, num_labels], trainable=False,
                                                              initializer=create_transition_matrix(self.extractor))

    def _train_eval(self):
        num_labels = self.extractor.vocab_size()
        seq_mask = None if constants.BERT_LENGTH_KEY in self.features else self.features.get(constants.SEQUENCE_MASK)
        self.loss = sequence_loss(logits=self.logits,
                                  targets=self.targets,
                                  sequence_lengths=self._sequence_lengths,
                                  num_labels=num_labels,
                                  crf=self.config.crf, tag_transitions=self._tag_transitions,
                                  label_smoothing=self.config.label_smoothing,
                                  confidence_penalty=self.config.confidence_penalty,
                                  mask=seq_mask)

        self.metric = tf.Variable(0, name=append_label(constants.OVERALL_KEY, self.name), dtype=tf.float32, trainable=False)

    def _eval_predict(self):
        predictions = crf.crf_decode(self.logits, self._tag_transitions, tf.cast(self._sequence_lengths, tf.int32))[0]
        # optionally mask intermediate subtokens from prediction results
        self.predictions = self._mask_subtokens(predictions)

    def _evaluation(self):
        self.evaluation_hooks = []
        self.metric_ops = {}
        predictions_key = append_label(constants.PREDICT_KEY, self.name)
        labels_key = append_label(constants.LABEL_KEY, self.name)

        eval_tensors = {  # tensors necessary for evaluation hooks (such as sequence length)
            constants.LENGTH_KEY: self._sequence_lengths,
            constants.SENTENCE_INDEX: self.features[constants.SENTENCE_INDEX],
            labels_key: self._mask_subtokens(self.targets),
            predictions_key: self.predictions,
        }

        overall_score = tf.identity(self.metric)
        self.metric_ops[append_label(constants.OVERALL_KEY, self.name)] = (overall_score, overall_score)
        overall_key = append_label(constants.OVERALL_KEY, self.name)
        # https://github.com/tensorflow/tensorflow/issues/20418 -- metrics don't accept variables, so we create a tensor
        eval_placeholder = tf.compat.v1.placeholder(dtype=tf.float32, name='update_%s' % overall_key)

        if constants.SRL_KEY in self.config.task:
            eval_tensors[constants.MARKER_KEY] = self.features[constants.MARKER_KEY]

            self.evaluation_hooks.append(
                SrlEvalHook(
                    tensors=eval_tensors,
                    evaluator=SrlEvaluator(
                        target=self.extractor,
                        output_path=os.path.join(self.params.job_dir, self.name + '.dev')),
                    label_key=labels_key,
                    predict_key=predictions_key,
                    eval_update=tf.compat.v1.assign(self.metric, eval_placeholder),
                    eval_placeholder=eval_placeholder,
                    output_confusions=self.params.verbose_eval,
                    output_dir=self.params.job_dir
                )
            )
        else:
            self.evaluation_hooks.append(
                SequenceEvalHook(
                    tensors=eval_tensors,
                    evaluator=TaggerEvaluator(
                        target=self.extractor,
                        output_path=os.path.join(self.params.job_dir, self.name + '.dev')),
                    label_key=labels_key,
                    predict_key=predictions_key,
                    eval_update=tf.compat.v1.assign(self.metric, eval_placeholder),
                    eval_placeholder=eval_placeholder,
                    output_dir=self.params.job_dir
                )
            )

    def _mask_subtokens(self, tensor_with_subtokens):
        mask = self.features.get(constants.SEQUENCE_MASK)
        if constants.BERT_LENGTH_KEY not in self.features and mask is not None:
            cond = tf.greater(mask, tf.zeros(tf.shape(input=mask), tf.int64))
            ignore = self.extractor.feat2index(BERT_SUBLABEL)
            tensor_with_subtokens = tf.compat.v1.where(cond,
                                                       tf.cast(tensor_with_subtokens, tf.int64),
                                                       tf.cast(tf.fill(tf.shape(input=tensor_with_subtokens), ignore), tf.int64))
            return tensor_with_subtokens
        return tensor_with_subtokens


class BiaffineSrlHead(TaggerHead):

    def __init__(self, inputs, config, features, params, training=False):
        super().__init__(inputs, config, features, params, training)
        self.n_steps = None
        self.predicate_indices = None

    def _all(self):
        inputs = get_encoder_input(self.inputs)
        input_shape = get_shape(inputs)  # (b x n x d), d == output_size
        self.n_steps, encoder_dim = input_shape[1], input_shape[2]

        # apply 2 arc and 2 rel MLPs to each output vector (1 for representing dependents, 1 for heads)
        def _mlp(size, name):
            return mlp(inputs, input_shape, self.config.mlp_dropout, size, self._training, name, n_splits=2)

        arg_mlp, predicate_mlp = _mlp(self.config.mlp_dim, name="rel_mlp")  # (bn x d), where d == rel_mlp_size

        # apply variable class biaffine classifier for semantic role labels
        with tf.compat.v1.variable_scope("bilinear_logits"):
            num_labels = self.extractor.vocab_size()  # r
            initializer = tf.compat.v1.zeros_initializer if self.config.zero_init else None
            self.logits = bilinear(arg_mlp, predicate_mlp, num_labels, self.n_steps, initializer=initializer)  # (b x n x r x n)

        if self.config.crf:
            # explicitly train a transition matrix
            self._tag_transitions = tf.compat.v1.get_variable("transitions", [num_labels, num_labels])
        else:
            # use constrained decoding based on IOB labels
            self._tag_transitions = tf.compat.v1.get_variable("transitions", [num_labels, num_labels], trainable=False,
                                                              initializer=create_transition_matrix(self.extractor))

        # batch-length vector of predicate indices
        predicate_indices = self.features[constants.PREDICATE_INDEX_KEY]
        predicate_indices = tf.expand_dims(predicate_indices, -1)
        # convert to [batch x n_steps] size Tensor, since each token's head is the predicate
        self.predicate_indices = tf.tile(predicate_indices, [1, self.n_steps])

    def _train_eval(self):
        self.mask = tf.sequence_mask(self.features[constants.LENGTH_KEY], name="padding_mask")

        num_labels = self.extractor.vocab_size()
        _logits = select_logits(self.logits, self.predicate_indices, self.n_steps)

        seq_mask = None if constants.BERT_LENGTH_KEY in self.features else self.features.get(constants.SEQUENCE_MASK)
        rel_loss = sequence_loss(logits=_logits,
                                 targets=self.targets,
                                 sequence_lengths=self._sequence_lengths,
                                 num_labels=num_labels,
                                 crf=self.config.crf,
                                 tag_transitions=self._tag_transitions,
                                 label_smoothing=self.config.label_smoothing,
                                 confidence_penalty=self.config.confidence_penalty, name="bilinear_loss",
                                 mask=seq_mask)

        self.loss = rel_loss
        self.metric = tf.Variable(0, name=append_label(constants.OVERALL_KEY, self.name), dtype=tf.float32, trainable=False)

    def _eval_predict(self):
        self.rel_probs = tf.nn.softmax(self.logits, axis=2)  # (b x n x r x n)
        self.n_tokens = tf.cast(tf.reduce_sum(input_tensor=self.features[constants.LENGTH_KEY]), tf.int32)
        _logits = select_logits(self.logits, self.predicate_indices, self.n_steps)  # (b x n x r)
        self.predictions = crf.crf_decode(_logits, self._tag_transitions, tf.cast(self._sequence_lengths, tf.int32))[0]
