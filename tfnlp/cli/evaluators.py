import os
from typing import List

import tensorflow as tf
from tfnlp.common import constants
from tfnlp.common.bert import BERT_SUBLABEL
from tfnlp.common.eval import append_srl_prediction_output, write_props_to_file, accuracy_eval
from tfnlp.common.eval import conll_eval, conll_srl_eval, parser_write_and_eval
from tfnlp.common.utils import binary_np_array_to_unicode


def get_evaluator(heads, feature_extractor, output_path, script_path):
    evaluators = {
        constants.TAGGER_KEY: TaggerEvaluator,
        constants.SRL_KEY: SrlEvaluator,
        constants.NER_KEY: TaggerEvaluator,
        constants.PARSER_KEY: DepParserEvaluator,
        constants.TOKEN_CLASSIFIER_KEY: TokenClassifierEvaluator
    }

    evals = []
    for head in heads:
        if head.task not in evaluators:
            raise ValueError("Unsupported head type: " + head.task)
        evaluator = evaluators[head.task](target=feature_extractor.targets[head.name],
                                          output_path=output_path + '.' + head.name,
                                          script_path=script_path)
        evals.append(evaluator)
    return AggregateEvaluator(evals)


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


class AggregateEvaluator(object):
    def __init__(self, evaluators: List[Evaluator]) -> None:
        super().__init__()
        self._evaluators = evaluators

    def __call__(self, labeled_instances, results):
        for evaluator in self._evaluators:
            evaluator(labeled_instances, results)


class TokenClassifierEvaluator(Evaluator):
    def __init__(self, target=None, output_path=None, script_path=None):
        super().__init__(target, output_path, script_path)

    def __call__(self, labeled_instances, results):
        target_key = constants.LABEL_KEY if not self.target.name else self.target.name
        labels = []
        gold = []
        indices = []
        for instance, result in zip(labeled_instances, results):
            labels.append(result[target_key].decode('utf-8'))
            gold.append(instance[self.target.key])
            indices.append(instance[constants.SENTENCE_INDEX])
        accuracy_eval(gold, labels, indices, output_file=self.output_path + '.txt')


class TaggerEvaluator(Evaluator):

    def __init__(self, target=None, output_path=None, script_path=None):
        super().__init__(target, output_path, script_path)

    def __call__(self, labeled_instances, results):
        target_key = constants.LABEL_KEY if not self.target.name else self.target.name
        labels_key = constants.LABEL_KEY if not self.target.key else self.target.key
        labels = []
        gold = []
        indices = []
        for instance, result in zip(labeled_instances, results):
            labels.append([label for label in binary_np_array_to_unicode(result[target_key]) if label != BERT_SUBLABEL])
            gold.append(instance[labels_key])
            indices.append(instance[constants.SENTENCE_INDEX])
        f1, result_str = conll_eval(gold, labels, indices, output_file=self.output_path + '.txt')
        tf.logging.info(result_str)


class DepParserEvaluator(Evaluator):
    def __init__(self, target=None, output_path=None, script_path=None):
        super().__init__(target, output_path, script_path)

    def __call__(self, labeled_instances, results):
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
                                       script_path=self.script_path,
                                       features=self.target,
                                       out_path=self.output_path + '.txt',
                                       gold_path=self.output_path + '.gold.txt')
        tf.logging.info('\n%s', result)


class SrlEvaluator(Evaluator):
    def __init__(self, target=None, output_path=None, script_path=None):
        super().__init__(target, output_path, script_path)

    def __call__(self, labeled_instances, results):
        labels = []
        gold = []
        markers = []
        indices = []
        for instance, result in zip(labeled_instances, results):
            labels.append([label for label in binary_np_array_to_unicode(result[constants.LABEL_KEY]) if label != BERT_SUBLABEL])
            gold.append(instance[constants.LABEL_KEY])
            markers.append(instance[constants.MARKER_KEY])
            indices.append(instance[constants.SENTENCE_INDEX])

        write_props_to_file(self.output_path + '.gold.txt', gold, markers, indices)
        write_props_to_file(self.output_path + '.txt', gold, markers, indices)

        result = conll_srl_eval(gold, labels, markers, indices)
        tf.logging.info(result)

        job_dir = os.path.dirname(self.output_path)

        # append results to summary file
        append_srl_prediction_output(os.path.basename(self.output_path), result, job_dir, output_confusions=True)
