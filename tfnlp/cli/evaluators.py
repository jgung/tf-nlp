import os
from collections import Counter

import tensorflow as tf
from tensorflow.python.lib.io import file_io

from tfnlp.common import constants
from tfnlp.common.chunk import start_of_chunk, end_of_chunk
from tfnlp.common.constants import LABEL_KEY, SENTENCE_INDEX
from tfnlp.common.eval import conll_eval, conll_srl_eval, write_props_to_file, SUMMARY_FILE, EVAL_LOG
from tfnlp.common.utils import binary_np_array_to_unicode


def get_evaluator(config):
    head_types = [head.type for head in config.heads]
    head_names = [head.name for head in config.heads]
    evaluators = {
        constants.TAGGER_KEY: tagger_evaluator,
        constants.SRL_KEY: srl_evaluator,
        constants.NER_KEY: tagger_evaluator,
    }

    if 'argtype' in head_names and LABEL_KEY in head_names and constants.SRL_KEY in head_types:
        return srl_ft_mtl_evaluator

    if head_types[0] not in evaluators:
        raise ValueError("Unsupported head type: " + head_types[0])
    return EvaluatorWrapper(evaluator=evaluators[head_types[0]], target=config.heads[0].name)


class EvaluatorWrapper(object):

    def __init__(self, evaluator, target):
        super().__init__()
        self.evaluator = evaluator
        self.target = target

    def __call__(self, labeled_instances, results, output_path=None):
        """
        Perform standard evaluation on a given list of gold labeled instances.
        :param labeled_instances: labeled instances
        :param results: prediction results corresponding to labeled instances
        :param output_path: path to output results to, or if none, use stdout
        """
        return self.evaluator(labeled_instances, results, output_path, target_key=self.target)


def tagger_evaluator(labeled_instances, results, output_path=None, target_key=None):
    target_key = LABEL_KEY if not target_key else target_key
    labels = []
    gold = []
    indices = []
    for instance, result in zip(labeled_instances, results):
        labels.append(binary_np_array_to_unicode(result[target_key]))
        gold.append(instance[LABEL_KEY])
        indices.append(instance[SENTENCE_INDEX])
    f1, result_str = conll_eval(gold, labels, indices, output_file=output_path)
    tf.logging.info(result_str)


def srl_evaluator(labeled_instances, results, output_path=None, target_key=None):
    labels = []
    gold = []
    markers = []
    indices = []
    for instance, result in zip(labeled_instances, results):
        labels.append(binary_np_array_to_unicode(result[target_key]))
        gold.append(instance[constants.LABEL_KEY])
        markers.append(instance[constants.MARKER_KEY])
        indices.append(instance[constants.SENTENCE_INDEX])

    write_props_to_file(output_path + '.gold.conll', gold, markers, indices)
    write_props_to_file(output_path + '.conll', gold, markers, indices)

    result = conll_srl_eval(gold, labels, markers, indices)
    tf.logging.info(result)
    p, r, f1 = result.evaluation.prec_rec_f1()

    job_dir = os.path.dirname(output_path)

    # append results to summary file
    summary_file = os.path.join(job_dir, SUMMARY_FILE)
    exists = tf.gfile.Exists(summary_file)
    with file_io.FileIO(summary_file, 'a') as summary:
        if not exists:
            summary.write('Path\t# Props\t% Perfect\tPrecision\tRecall\tF1\n')
        summary.write('%s\t%d\t%f\t%f\t%f\t%f\n' % (os.path.basename(output_path),
                                                    result.ntargets,
                                                    result.perfect_props(),
                                                    p, r, f1))

    # append evaluation log
    with file_io.FileIO(os.path.join(job_dir, EVAL_LOG), 'a') as eval_log:
        eval_log.write('\n%s\n' % os.path.basename(output_path))
        eval_log.write(str(result) + '\n')
        eval_log.write('\n%s\n\n' % result.confusion_matrix())


def srl_ft_mtl_evaluator(labeled_instances, results, output_path=None, target_key=None, type_key='argtype'):
    # predicted/gold function tag label
    predicted_labels, gold_labels = [], []
    # predicted/gold core vs. adjunct
    predicted_types, gold_types = [], []
    # labels automatically converted back with core-argument distinction
    combined, combined_gold = [], []
    # predicate markers
    markers = []
    # sentence indices to preserve order
    indices = []
    for instance, result in zip(labeled_instances, results):
        labels = binary_np_array_to_unicode(result[target_key])
        types = binary_np_array_to_unicode(result[type_key])
        predicted_labels.append(labels)
        predicted_types.append(types)
        combined.append(_convert_core_vs_adjunct(labels, types))

        gold_labels.append(instance[constants.LABEL_KEY])
        gold_types.append(instance[type_key])
        combined_gold.append(_convert_core_vs_adjunct(instance[constants.LABEL_KEY], instance[type_key]))

        markers.append(instance[constants.MARKER_KEY])
        indices.append(instance[constants.SENTENCE_INDEX])

    labels_result = conll_srl_eval(gold_labels, predicted_labels, markers, indices)
    types_result = conll_srl_eval(gold_types, predicted_types, markers, indices)
    combined_result = conll_srl_eval(combined_gold, combined, markers, indices)
    tf.logging.info('Labels results:\n%s' % str(labels_result))
    tf.logging.info('Types results:\n%s' % str(types_result))
    tf.logging.info('Combined results:\n%s' % str(combined_result))


def _convert_core_vs_adjunct(labels, argtypes):
    """
    Given a list of role labels and a corresponding list of types, convert to a single list of labels preserving argument
    adjunct distinction. Determine whether a given argument is core or not based on count of core labels within span.
    :param labels: role labels
    :param argtypes: role label types, adjunct vs. core (ARGM vs. ARG)
    :return: list of converted labels
    """

    def _is_adjunct(_argtype):
        return 'ARGM' in _argtype

    def _convert(_label, _argtype):
        if _label == 'O':
            return _label
        return _label + '-' + _argtype

    result = []
    curr_chunk = []
    argtype_counter = Counter()
    prev_label = None
    for curr_label, argtype in zip(labels, argtypes):
        if end_of_chunk(prev_label, curr_label):
            # convert chunk to core vs. adjunct
            chunk_arg_type = 'ARG' if argtype_counter['ARG'] >= argtype_counter['ARGM'] else 'ARGM'
            curr_chunk = [_convert(l, chunk_arg_type) for l in curr_chunk]

            result.extend(curr_chunk)
            curr_chunk = []
            argtype_counter = Counter()
        elif curr_chunk:
            curr_chunk.append(curr_label)

        if start_of_chunk(prev_label, curr_label):
            curr_chunk.append(curr_label)

        if curr_chunk:
            if _is_adjunct(argtype):
                argtype_counter['ARGM'] += 1
            else:
                argtype_counter['ARG'] += 1

        prev_label = curr_label
    if curr_chunk:
        chunk_arg_type = argtype_counter.most_common(1)[0][0]
        curr_chunk = [_convert(l, chunk_arg_type) for l in curr_chunk]
        result.extend(curr_chunk)

    return result
