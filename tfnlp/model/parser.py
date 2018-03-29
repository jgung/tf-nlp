import tensorflow as tf
from tensorflow.python.estimator.export.export_output import PredictOutput
from tensorflow.python.saved_model import signature_constants

from tfnlp.common.config import get_gradient_clip, get_optimizer
from tfnlp.common.constants import ARC_PROBS, DEPREL_KEY, HEAD_KEY, LABELED_ATTACHMENT_SCORE, LABEL_SCORE, LENGTH_KEY, \
    REL_PROBS, UNLABELED_ATTACHMENT_SCORE, WORD_KEY
from tfnlp.common.eval import ParserEvalHook, log_trainable_variables
from tfnlp.layers.layers import encoder, input_layer


def parser_model_func(features, mode, params):
    inputs = input_layer(features, params, mode == tf.estimator.ModeKeys.TRAIN)

    outputs = encoder(features, inputs, mode, params)

    outputs = tf.concat(values=outputs, axis=-1)
    time_steps = tf.shape(outputs)[1]
    rnn_outputs = tf.reshape(outputs, [-1, params.config.state_size * 2], name="flatten_rnn_outputs_for_linear_projection")

    def mlp(size, name):
        result = tf.layers.dense(rnn_outputs, size, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer,
                                 name=name)
        return tf.layers.dropout(result, params.config.mlp_dropout, training=mode == tf.estimator.ModeKeys.TRAIN)

    arc_mlp_size, rel_mlp_size = 500, 100
    dep_arc_mlp = mlp(arc_mlp_size, name="dep_arc_mlp")  # produces (batch_size * seq_len) x arc_mlp_size matrix
    head_arc_mlp = mlp(arc_mlp_size, name="head_arc_mlp")
    dep_rel_mlp = mlp(rel_mlp_size, name="dep_rel_mlp")  # produces (batch_size * seq_len) x rel_mlp_size matrix
    head_rel_mlp = mlp(rel_mlp_size, name="head_rel_mlp")

    with tf.variable_scope("arc_bilinear_logits"):
        arc_logits = bilinear(dep_arc_mlp, head_arc_mlp, 1, time_steps, include_bias1=True)
        arc_predictions = tf.argmax(arc_logits, axis=-1)

    with tf.variable_scope("rel_bilinear_logits"):
        target = params.extractor.targets[DEPREL_KEY]
        num_labels = target.vocab_size()
        rel_logits = bilinear(dep_rel_mlp, head_rel_mlp, num_labels, time_steps, include_bias1=True, include_bias2=True)

        one_hot = tf.one_hot(arc_predictions, time_steps)
        select_rel_logits = tf.matmul(rel_logits, tf.expand_dims(one_hot, -1))
        select_rel_logits = tf.squeeze(select_rel_logits, axis=-1)

    loss = None
    train_op = None
    eval_metric_ops = None
    export_outputs = None
    evaluation_hooks = None
    rel_predictions = None
    arc_probs = None
    rel_probs = None
    arc_targets = None
    rel_targets = None
    n_tokens = None

    mask = tf.sequence_mask(features[LENGTH_KEY], name="padding_mask")
    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        log_trainable_variables()

        with tf.variable_scope("arc_bilinear_loss"):
            arc_targets = tf.identity(features[HEAD_KEY], name=HEAD_KEY)
            arc_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=arc_logits, labels=arc_targets)
            arc_losses = tf.boolean_mask(arc_losses, mask, name="mask_padding_from_loss")
            arc_loss = tf.reduce_mean(arc_losses)

        with tf.variable_scope("rel_bilinear_loss"):
            rel_targets = tf.identity(features[DEPREL_KEY], name=DEPREL_KEY)
            rel_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=select_rel_logits, labels=rel_targets)
            rel_losses = tf.boolean_mask(rel_losses, mask, name="mask_padding_from_loss")
            rel_loss = tf.reduce_mean(rel_losses)
        loss = arc_loss + rel_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = get_optimizer(params.config)
        parameters = tf.trainable_variables()
        gradients = tf.gradients(loss, parameters)
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=get_gradient_clip(params.config))
        train_op = optimizer.apply_gradients(zip(gradients, parameters), global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
        arc_probs = tf.nn.softmax(arc_logits)
        rel_probs = tf.nn.softmax(rel_logits, axis=2)
        n_tokens = tf.cast(tf.reduce_sum(features[LENGTH_KEY]), tf.int32)
        rel_predictions = tf.argmax(select_rel_logits, axis=-1)

    if mode == tf.estimator.ModeKeys.EVAL:
        arc_correct = tf.boolean_mask(tf.to_int32(tf.equal(arc_predictions, arc_targets)), mask)
        rel_correct = tf.boolean_mask(tf.to_int32(tf.equal(rel_predictions, rel_targets)), mask)
        n_arc_correct = tf.cast(tf.reduce_sum(arc_correct), tf.int32)
        n_rel_correct = tf.cast(tf.reduce_sum(rel_correct), tf.int32)
        correct = arc_correct * rel_correct
        n_correct = tf.cast(tf.reduce_sum(correct), tf.int32)
        eval_metric_ops = {UNLABELED_ATTACHMENT_SCORE: tf.metrics.mean(n_arc_correct / n_tokens),
                           LABEL_SCORE: tf.metrics.mean(n_rel_correct / n_tokens),
                           LABELED_ATTACHMENT_SCORE: tf.metrics.mean(n_correct / n_tokens)}
        evaluation_hooks = [ParserEvalHook(
            {ARC_PROBS: arc_probs, REL_PROBS: rel_probs, WORD_KEY: features[WORD_KEY], LENGTH_KEY: features[LENGTH_KEY],
             HEAD_KEY: features[HEAD_KEY], DEPREL_KEY: features[DEPREL_KEY]},
            features=params.extractor, script_path=params.script_path, )]

    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: PredictOutput(arc_probs),
                          DEPREL_KEY: PredictOutput(rel_probs)}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions={DEPREL_KEY: rel_predictions, HEAD_KEY: arc_predictions,
                                                   ARC_PROBS: arc_probs, REL_PROBS: rel_probs},
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops,
                                      export_outputs=export_outputs,
                                      evaluation_hooks=evaluation_hooks)


def bilinear(input1, input2, output_size, timesteps, include_bias1=False, include_bias2=False):
    input1dim = input1.get_shape()[-1]
    input2dim = input2.get_shape()[-1]
    if input1dim != input2dim:
        raise ValueError("Last dimensions of input1 and input2 must match: got %s and %s" % (input1dim, input2dim))
    if include_bias1:
        input1dim += 1
        input1 = tf.concat([input1, tf.ones([tf.shape(input1)[0], 1])], -1)
    if include_bias2:
        input2dim += 1
        input2 = tf.concat([input2, tf.ones([tf.shape(input2)[0], 1])], -1)

    weights = tf.get_variable("weights", shape=[input1dim, output_size, input2dim], initializer=tf.orthogonal_initializer)
    # (b n x d) (d x r d) -> b n x r d
    linear_mapping = tf.matmul(tf.reshape(input1, [-1, input1dim]), tf.reshape(weights, [input1dim, -1]))
    # (b x n r x d) (b x n x d)T -> b x n x n r (batch matrix multiplication)
    bilinear_result = tf.matmul(tf.reshape(linear_mapping, [-1, output_size * timesteps, input2dim]),
                                tf.reshape(input2, [-1, timesteps, input2dim]), transpose_b=True)
    if output_size == 1:
        return tf.reshape(bilinear_result, [-1, timesteps, timesteps])
    return tf.reshape(bilinear_result, [-1, timesteps, output_size, timesteps])
