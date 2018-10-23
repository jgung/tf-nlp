import tempfile
import unittest

import pkg_resources
from tensorflow import Session

from tfnlp.common.constants import CHAR_KEY, LABEL_KEY, LENGTH_KEY, WORD_KEY
from tfnlp.common.utils import read_json
from tfnlp.feature import get_feature_extractor


def test_extractor():
    configpath = pkg_resources.resource_filename(__name__, "resources/feats.json")
    config = read_json(configpath)
    extractor = get_feature_extractor(config.features)
    extractor.initialize()
    extractor.train()
    return extractor


class TestFeature(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.extractor = test_extractor()
        self.sentence = {LABEL_KEY: '0', WORD_KEY: "the cat sat on the mat".split()}
        self.other_sentence = {LABEL_KEY: '1', WORD_KEY: "the foo".split()}

    def test_scalar(self):
        feats = self.extractor.extract(self.sentence)
        self.assertEqual(1, feats.context.feature[LABEL_KEY].int64_list.value[0])

    def test_length(self):
        feats = self.extractor.extract(self.sentence)
        self.assertEqual(6, feats.context.feature[LENGTH_KEY].int64_list.value[0])

    def test_sequence(self):
        feats = self.extractor.extract(self.sentence)
        self.assertEqual([[2], [3], [4], [5], [2], [6]],
                         [feat.int64_list.value for feat in feats.feature_lists.feature_list[WORD_KEY].feature])

    def test_sequence_list(self):
        feats = self.extractor.extract(self.sentence)
        char_feature = self.extractor.feature(CHAR_KEY)
        left_padding = char_feature.left_padding * [char_feature.indices[char_feature.left_pad_word]]
        right_padding = char_feature.right_padding * [char_feature.indices[char_feature.right_pad_word]]
        pad_index = char_feature.indices.get(char_feature.pad_word)
        padding = (char_feature.max_len - 3 - len(left_padding) - len(right_padding)) * [pad_index]

        self.assertEqual(6, len(feats.feature_lists.feature_list[CHAR_KEY].feature))
        self.assertEqual(left_padding + [4, 5, 6] + right_padding + padding,
                         feats.feature_lists.feature_list[CHAR_KEY].feature[0].int64_list.value)
        self.assertEqual(left_padding + [12, 8, 4] + right_padding + padding,
                         feats.feature_lists.feature_list[CHAR_KEY].feature[5].int64_list.value)

    def test_not_train(self):
        self.extractor.extract(self.sentence)
        self.extractor.test()
        feats = self.extractor.extract(self.other_sentence)
        word_feature = self.extractor.feature(WORD_KEY)
        self.assertEqual([[2], [word_feature.indices.get(word_feature.unknown_word)]],
                         [feat.int64_list.value for feat in feats.feature_lists.feature_list[WORD_KEY].feature])

    def test_parse(self):
        example = self.extractor.extract(self.sentence)
        result = self.extractor.parse(example.SerializeToString())

        char_feature = self.extractor.feature(CHAR_KEY)
        self.assertEqual(5, len(result))
        self.assertEqual(char_feature.max_len, result[CHAR_KEY].shape.dims[1].value)
        with Session():
            result[CHAR_KEY].eval()
            result[WORD_KEY].eval()
            result[LABEL_KEY].eval()
            result[LENGTH_KEY].eval()

    def test_read_vocab(self):
        dirpath = pkg_resources.resource_filename(__name__, "resources/vocab/word.txt")
        word_feature = self.extractor.feature(WORD_KEY)
        word_feature.read_vocab(dirpath)
        # 0     1   2   3   0   4
        # the   cat sat on  the mat
        self.assertEqual(7, len(word_feature.indices))
        self.assertEqual("the", word_feature.index_to_feat(0))
        self.assertEqual("mat", word_feature.index_to_feat(4))

    def test_write_and_read_vocab(self):
        self.extractor.extract(self.sentence)
        file = tempfile.NamedTemporaryFile()
        word_feature = self.extractor.feature(WORD_KEY)
        word_feature.write_vocab(file.name, overwrite=True, prune=True)
        word_feature.read_vocab(file.name)
        self.assertEqual(7, len(word_feature.indices))
        self.assertEqual("mat", word_feature.index_to_feat(6))

    def test_read_config(self):
        extractor = test_extractor()
        feats = extractor.extract(self.sentence)
        self.assertEqual(6, len(feats.feature_lists.feature_list[CHAR_KEY].feature))
        self.assertEqual([[2], [3], [4], [5], [2], [6]],
                         [feat.int64_list.value for feat in feats.feature_lists.feature_list[WORD_KEY].feature])
        self.assertEqual([1], feats.context.feature[LABEL_KEY].int64_list.value)