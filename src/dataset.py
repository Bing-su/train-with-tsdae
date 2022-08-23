from typing import Callable, Optional

import nltk
import numpy as np
from datasets import IterableDataset as dsIterableDataset
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sentence_transformers.readers.InputExample import InputExample
from torch.utils.data import IterableDataset


class DenoisingAutoEncoderIterableDataset(IterableDataset):
    """
    The DenoisingAutoEncoderDataset returns InputExamples in the format:
        texts=[noise_fn(sentence), sentence]
    It is used in combination with the DenoisingAutoEncoderLoss:
        Here, a decoder tries to re-construct the sentence without noise.
    :param dataset: A huggingface datasets streaming dataset
    :param column_name: The name of the column in the dataset
                        that contains the sentences
    :param noise_fn: A noise function:
                     Given a string, it returns a string with noise, e.g. deleted words
    :param del_ratio: The ratio of words to delete, 0 <= del_ratio <= 1
    """

    def __init__(
        self,
        dataset: dsIterableDataset,
        column_name: str = "text",
        noise_fn: Optional[Callable[[str], str]] = None,
        del_ratio: float = 0.6,
    ):
        self.dataset = dataset
        self.column_name = column_name
        if noise_fn is None:
            self.noise_fn = self.delete
        else:
            self.noise_fn = noise_fn

        self.detokenizer = TreebankWordDetokenizer()
        self.del_ratio = del_ratio

    def __iter__(self):
        for data in self.dataset:
            sent = data[self.column_name]
            yield InputExample(texts=[self.noise_fn(sent), sent])

    def __len__(self):
        return self.dataset.info.splits["train"].num_examples

    # Deletion noise.
    def delete(self, text: str):
        words = nltk.word_tokenize(text)
        n = len(words)
        if n == 0:
            return text

        keep_or_not = np.random.rand(n) > self.del_ratio
        if sum(keep_or_not) == 0:
            keep_or_not[
                np.random.choice(n)
            ] = True  # guarantee that at least one word remains
        words_processed = self.detokenizer.detokenize(np.array(words)[keep_or_not])
        return words_processed
