import os

import tensorflow as tf

from tfnlp.common import constants
from tfnlp.common.bert import BERT_SUBLABEL
from tfnlp.common.eval import append_srl_prediction_output, write_props_to_file, apply_srl_mappings, convert_to_original
from tfnlp.common.eval import conll_eval, conll_srl_eval, parser_write_and_eval
from tfnlp.common.utils import binary_np_array_to_unicode


def get_evaluator(config, feature_extractor, output_path, script_path):
    head = config.heads[0]

    evaluators = {
        constants.TAGGER_KEY: TaggerEvaluator,
        constants.SRL_KEY: SrlEvaluator,
        constants.NER_KEY: TaggerEvaluator,
        constants.PARSER_KEY: DepParserEvaluator,
    }

    if head.task in evaluators:
        evaluator = evaluators[head.task](target=feature_extractor.targets[head.name],
                                          output_path=output_path,
                                          script_path=script_path)
    elif head.task == constants.SRL_FT_KEY:
        evaluator = SrlGroupedEvaluator(target=feature_extractor.targets[head.name],
                                        output_path=output_path,
                                        script_path=script_path,
                                        mappings=head.mappings)
    else:
        raise ValueError("Unsupported head type: " + head.task)
    return evaluator


class Evaluator(object):

    def __init__(self, target=None, output_path=None, script_path=None):
        super().__init__()
        self.target = target
        self.output_path = output_path
        self.script_path = script_path

    def __call__(self, labeled_instances, results):
        """
        Perform standard evaluation on a given list of gold labeled instances.
        :param labeled_instances: labeled instances
        :param results: prediction results corresponding to labeled instances
        """
        pass


class TaggerEvaluator(Evaluator):

    def __init__(self, target=None, output_path=None, script_path=None):
        super().__init__(target, output_path, script_path)

    def __call__(self, labeled_instances, results):
        tagger_evaluator(labeled_instances, results, self.output_path, self.target.name)


def tagger_evaluator(labeled_instances, results, output_path=None, target_key=None):
    target_key = constants.LABEL_KEY if not target_key else target_key
    labels = []
    gold = []
    indices = []
    for instance, result in zip(labeled_instances, results):
        labels.append([label for label in binary_np_array_to_unicode(result[target_key]) if label != BERT_SUBLABEL])
        gold.append(instance[constants.LABEL_KEY])
        indices.append(instance[constants.SENTENCE_INDEX])
    f1, result_str = conll_eval(gold, labels, indices, output_file=output_path)
    tf.logging.info(result_str)


class DepParserEvaluator(Evaluator):
    def __init__(self, target=None, output_path=None, script_path=None):
        super().__init__(target, output_path, script_path)

    def __call__(self, labeled_instances, results):
        dep_evaluator(labeled_instances, results, self.target, self.script_path, self.output_path)


def dep_evaluator(labeled_instances, results, features, script_path, output_path=None):
    arc_probs = []
    rel_probs = []
    gold_arcs = []
    gold_rels = []

    for instance, result in zip(labeled_instances, results):
        seq_len = 1 + len(instance[constants.WORD_KEY])  # plus 1 for head
        gold_arcs.append([0] + instance[constants.HEAD_KEY])
        gold_rels.append(['<ROOT>'] + instance[constants.DEPREL_KEY])

        arc_probs.append(result[constants.ARC_PROBS][:seq_len, :seq_len])
        rel_probs.append(result[constants.REL_PROBS])

    result = parser_write_and_eval(arc_probs=arc_probs,
                                   rel_probs=rel_probs,
                                   heads=gold_arcs,
                                   rels=gold_rels,
                                   script_path=script_path,
                                   features=features,
                                   out_path=output_path,
                                   gold_path=output_path + '.gold')
    tf.logging.info('\n%s', result)


class SrlEvaluator(Evaluator):
    def __init__(self, target=None, output_path=None, script_path=None):
        super().__init__(target, output_path, script_path)

    def __call__(self, labeled_instances, results):
        srl_evaluator(labeled_instances, results, self.output_path, constants.LABEL_KEY)


def srl_evaluator(labeled_instances, results, output_path=None, target_key=None):
    labels = []
    gold = []
    markers = []
    indices = []
    for instance, result in zip(labeled_instances, results):
        labels.append([label for label in binary_np_array_to_unicode(result[target_key]) if label != BERT_SUBLABEL])
        gold.append(instance[constants.LABEL_KEY])
        markers.append(instance[constants.MARKER_KEY])
        indices.append(instance[constants.SENTENCE_INDEX])

    write_props_to_file(output_path + '.gold.conll', gold, markers, indices)
    write_props_to_file(output_path + '.conll', gold, markers, indices)

    result = conll_srl_eval(gold, labels, markers, indices)
    tf.logging.info(result)

    job_dir = os.path.dirname(output_path)

    # append results to summary file
    append_srl_prediction_output(os.path.basename(output_path), result, job_dir, output_confusions=True)


class SrlGroupedEvaluator(Evaluator):
    def __init__(self, mappings, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mappings = mappings

    def __call__(self, labeled_instances, results):
        srl_grouped_evaluator(labeled_instances, results, self.mappings, self.output_path, constants.LABEL_KEY)


def srl_grouped_evaluator(labeled_instances, results, mappings, output_path, target_key=constants.LABEL_KEY):
    # predicted/gold function tag label
    predicted_labels, gold_labels, original_labels = [], [], []
    # predicate markers
    markers = []
    # sentence indices to preserve order
    indices = []
    for instance, result in zip(labeled_instances, results):
        labels = binary_np_array_to_unicode(result[target_key])
        predicted_labels.append(labels)

        gold_labels.append(instance[constants.LABEL_KEY])
        original_labels.append(instance['original'])

        markers.append(instance[constants.MARKER_KEY])
        indices.append(instance[constants.SENTENCE_INDEX])

    def evaluate(gold, predicted, tag):
        write_props_to_file('%s.%s.gold.conll' % (output_path, tag), gold, markers, indices)
        write_props_to_file('%s.%s.conll' % (output_path, tag), predicted, markers, indices)
        res = conll_srl_eval(gold, predicted, markers, indices)
        tf.logging.info('%s results:\n%s' % (tag, str(res)))

    # evaluate on mapped label set
    evaluate(gold_labels, predicted_labels, 'training')

    # now run evaluation on original label set (still split by FT)
    predicted_labels = apply_srl_mappings(predicted_labels, original_labels, mappings)
    evaluate(original_labels, predicted_labels, 'split')

    # evaluate on original labels, use result for early stopping
    original_labels = convert_to_original(original_labels)
    predicted_labels = convert_to_original(predicted_labels)
    evaluate(original_labels, predicted_labels, 'original')
