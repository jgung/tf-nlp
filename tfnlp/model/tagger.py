import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.python.estimator.export.export_output import PredictOutput
from tensorflow.python.ops.lookup_ops import index_to_string_table_from_file

from tfnlp.common.config import get_gradient_clip, get_optimizer
from tfnlp.common.constants import ACCURACY_METRIC_KEY, LABEL_KEY, LENGTH_KEY, PREDICT_KEY
from tfnlp.common.eval import SequenceEvalHook, log_trainable_variables
from tfnlp.common.metrics import tagger_metrics
from tfnlp.layers.layers import encoder, input_layer

TRANSITIONS = "transitions"
SCORES_KEY = "scores"


def model_func(features, mode, params):
    inputs = input_layer(features, params, mode == tf.estimator.ModeKeys.TRAIN)
    outputs = encoder(features, inputs, mode, params)

    outputs = tf.concat(values=outputs, axis=-1)
    time_steps = tf.shape(outputs)[1]
    rnn_outputs = tf.reshape(outputs, [-1, params.config.state_size * 2], name="flatten_rnn_outputs_for_linear_projection")

    target = params.extractor.targets[LABEL_KEY]
    num_labels = target.vocab_size()
    logits = tf.reshape(tf.layers.dense(rnn_outputs, num_labels), [-1, time_steps, num_labels], name="unflatten_logits")
    transition_matrix = tf.get_variable(TRANSITIONS, [num_labels, num_labels])

    targets = None
    predictions = None
    loss = None
    train_op = None
    eval_metric_ops = None
    export_outputs = None
    evaluation_hooks = None

    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        log_trainable_variables()
        targets = tf.identity(features[LABEL_KEY], name=LABEL_KEY)

        if params.config.crf:
            log_likelihood, _ = crf_log_likelihood(logits, targets, sequence_lengths=features[LENGTH_KEY],
                                                   transition_params=transition_matrix)
            losses = -log_likelihood
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
            mask = tf.sequence_mask(features[LENGTH_KEY], name="padding_mask")
            losses = tf.boolean_mask(losses, mask, name="mask_padding_from_loss")
        loss = tf.reduce_mean(losses)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = get_optimizer(params.config)
        parameters = tf.trainable_variables()
        gradients = tf.gradients(loss, parameters)
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=get_gradient_clip(params.config))
        train_op = optimizer.apply_gradients(zip(gradients, parameters), global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
        predictions, _ = tf.contrib.crf.crf_decode(logits, transition_matrix, features[LENGTH_KEY])

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = tagger_metrics(predictions=tf.cast(predictions, dtype=tf.int64), labels=targets)
        eval_metric_ops[ACCURACY_METRIC_KEY] = tf.metrics.accuracy(labels=targets, predictions=predictions)
        evaluation_hooks = [SequenceEvalHook(script_path=params.script_path,
                                             gold_tensor=targets,
                                             predict_tensor=predictions,
                                             length_tensor=features[LENGTH_KEY],
                                             vocab=params.extractor.targets[LABEL_KEY])]

    if mode == tf.estimator.ModeKeys.PREDICT:
        index_to_label = index_to_string_table_from_file(vocabulary_file=params.label_vocab_path,
                                                         default_value=target.index_to_feat(target.unknown_index))
        predictions = index_to_label.lookup(tf.cast(predictions, dtype=tf.int64))
        export_outputs = {PREDICT_KEY: PredictOutput(predictions)}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops,
                                      export_outputs=export_outputs,
                                      evaluation_hooks=evaluation_hooks)