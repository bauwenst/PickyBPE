from __future__ import annotations

from typing import Union, Iterable, TypeVar
from enum import Enum
from pathlib import Path

from collections import defaultdict
import numpy as np
import time
import json
import logging
logger = logging.getLogger(__name__)

from .utils import MCounter, WHITESPACE, PAD, UNK, BOS, EOS, Token, Word


T = TypeVar("T")
def modlog(iterable: Iterable[T], step: int, units: str="elements", message: str="Processed") -> Iterable[T]:
    """
    Log a message each time after iterating over a fixed amount of elements in an iterable.
    """
    for i, thing in enumerate(iterable):
        yield thing
        if i > 0 and i % step == 0:
            logger.info(f"{message} {i} {units}.")


class EventType(Enum):
    MERGE = 0
    SPLIT = 1


class PickyBPETrainer:

    def __init__(
        self,
        vocab_size: int,
        character_coverage: float = 0.9999,
        picky_threshold: float = 0.9999,

        include_specials: bool = True,
        pad_id: int = 0,
        unk_id: int = 1,
        bos_id: int = 2,
        eos_id: int = 3,
    ):
        self.desired_vocab_size = vocab_size
        self.coverage: float = character_coverage
        self.threshold: float = picky_threshold

        if include_specials:
            self.pad_token = Token(pad_id, PAD, 0, special=True)
            self.unk_token = Token(unk_id, UNK, 0, special=True)
            self.bos_token = Token(bos_id, BOS, 0, special=True)
            self.eos_token = Token(eos_id, EOS, 0, special=True)
            specials = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
            max_id = max(token.id for token in specials)
        else:
            specials = []
            max_id = -1

        self.id2token  = {token.id: token  for token in specials}
        self.str2token = {token.str: token for token in specials}
        self.str2token = defaultdict(lambda: self.unk_token, self.str2token)
        self.actual_vocab_size = len(specials)
        self.new_id            = max_id + 1

        self.events: list[Union[tuple[EventType,Token,list[Token]],tuple[EventType,list[Token],Token]]] = []

    def _count_words_in_file(self, file: str) -> MCounter:
        counter = MCounter()

        logger.info(f'Loading corpus from {file}...')
        start_time = time.time()
        num_lines = 0
        with open(file, "r", encoding="utf-8") as handle:
            for line in modlog(handle, 500_000, "lines"):
                counter.update(line.strip('\n').split())
                num_lines += 1

        logger.info(f'Loaded {len(counter)} unique words from {num_lines} sentences in {time.time() - start_time:.2f}s.')
        return counter

    def _string_to_atoms(self, word: str) -> Iterable[str]:
        return WHITESPACE + word

    def _pretokens_to_objects(self, pretoken_counts: MCounter) -> list[Word]:
        return [Word(i, self._string_to_atoms(pretoken), freq) for i, (pretoken, freq) in enumerate(pretoken_counts.items())]

    def _count_atoms(self, words: list[Word]) -> MCounter:
        counter = MCounter()
        for word in modlog(words, 500_000, "words"):
            counter.update(MCounter(word.atoms) * word.freq)
        return counter

    def _filter_atoms(self, characters: MCounter) -> MCounter:
        if self.coverage < 1:
            corpus_size = sum(characters.values())
            freq_to_remove = corpus_size - round(self.coverage * corpus_size)
            if freq_to_remove > 0:
                cum_sum = np.cumsum([freq for _, freq in reversed(characters.most_common())])
                num_to_remove = np.searchsorted(cum_sum, freq_to_remove)
                characters_to_remove = [c for c, _ in characters.most_common()[-num_to_remove:]]
                for c in characters_to_remove:
                    characters.pop(c)
                logger.info(f'Replaced {num_to_remove} rare characters with UNK.')
        return characters

    def _initialize_vocab(self, words: list[Word]):
        logger.info('Initializing the vocabulary...')
        filtered_characters = self._filter_atoms(self._count_atoms(words))
        for i, character in enumerate(filtered_characters):
            token = Token(self.new_id + i, character, filtered_characters[character])
            self.id2token[token.id]   = token
            self.str2token[token.str] = token

        self.new_id            += len(filtered_characters)
        self.actual_vocab_size += len(filtered_characters)
        logger.info(f'Initialized vocabulary with {len(filtered_characters)} unique characters.')

    @staticmethod
    def _validate_pair(pair: np.ndarray) -> bool:
        return not any(token.special for token in pair)

    def _encode_words(self, words: list[Word]):
        logger.info("Encoding words...")
        for word in modlog(words, 500_000, "words"):
            word.initialize_tokens(self.str2token)  # Stores the resulting tokenisation in-place, and links each token to the words that contain it.

    def _initialize_pairs(self, words: list[Word]) -> MCounter:
        pairs = MCounter()
        logger.info("Counting character pairs...")
        for word in modlog(words, 500_000, "words"):
            pairs.update(word.pairs)

        to_remove = set()
        for pair in pairs:
            if not self._validate_pair(pair):
                to_remove.add(pair)
        for pair in to_remove:
            pairs.pop(pair)
        return pairs

    @staticmethod
    def _update_pairs_on_merge(new_token: Token, pair: tuple[Token, Token], pairs_for_update: MCounter, pairs: MCounter):
        pairs.update(pairs_for_update)
        for p, freq in pairs_for_update.items():
            if new_token not in p:
                raise ValueError(f'Pair {p} does not contain the new token {new_token}.')
            if new_token is p[0]:
                if new_token is p[1]:
                    to_update = (pair[1], pair[0])
                else:
                    to_update = (pair[1], p[1])
            else:
                to_update = (p[0], pair[0])
            if to_update in pairs:
                pairs[to_update] -= freq
                if pairs[to_update] <= 0:
                    pairs.pop(to_update)

    @staticmethod
    def _update_pairs_on_remove(token: Token, split: list[Token], pairs_for_update: MCounter, pairs: MCounter):
        for pair, freq in pairs_for_update.items():
            if token is pair[0]:
                if token is pair[1]:
                    to_update = (split[-1], split[0])
                else:
                    to_update = (split[-1], pair[1])
            else:
                to_update = (pair[0], split[0])
            pairs[to_update] += freq
            pairs.pop(pair)

    def _remove_if_possible(self, token: Token, merged_freq: int, pairs: MCounter) -> bool:
        if merged_freq / (token.freq + merged_freq) > self.threshold:
            split = token.split_if_possible()
            if split is not None:
                self.actual_vocab_size -= 1
                for t in split:
                    t.freq += token.freq
                for pair in zip(split[:-1], split[1:]):
                    pairs[pair] += token.freq
                pairs_for_update = MCounter()
                for word in token.words:
                    if token not in word.tokens:
                        raise ValueError(f'Token {token} not found in the token list {word.tokens} of word {word}.')
                    pairs_for_update.update({pair: freq for pair, freq in word.pairs.items() if
                                            self._validate_pair(pair) and token in pair})
                    word.split_token(token, split)
                self._update_pairs_on_remove(token, split, pairs_for_update, pairs)
                token.remove()
                return True
        return False

    def _merge_token_in_words(self, token_to_merge: Token, pair_to_merge: tuple[Token, Token], pairs: MCounter) -> int:
        # Find how many times the pair appears, and where.
        actual_freq = 0
        pairs_for_update = MCounter()
        for word in pair_to_merge[0].words & pair_to_merge[1].words:
            if pair_to_merge in word.pairs:
                word.pairs.pop(pair_to_merge)
                actual_freq += word.merge_pair(pair_to_merge, token_to_merge)
                pairs_for_update.update(
                    {p: f for p, f in word.pairs.items() if self._validate_pair(p) and token_to_merge in p}
                )
        self._update_pairs_on_merge(token_to_merge, pair_to_merge, pairs_for_update, pairs)
        token_to_merge.freq += actual_freq

        # Update the tokens being merged, and possibly PickyBPE them.
        for token in set(pair_to_merge):
            if not token.present:
                raise ValueError(f'Token {token} is not present in the vocabulary.')
            token.freq -= actual_freq * pair_to_merge.count(token)

            # PickyBPE
            remaining_token_freq = token.freq
            removed = self._remove_if_possible(token, actual_freq, pairs)
            if removed:  # token.freq == 0
                logger.info(f'Removed token {token.str} with frequency {remaining_token_freq} after merging into {token_to_merge.str} with frequency {token_to_merge.freq}.')
                self.events.append((EventType.SPLIT, token, token.walk()))

        return actual_freq

    def _merge_pair(self, pair: tuple[Token, Token], pairs: MCounter) -> int:
        pairs.pop(pair)
        merged_str = pair[0].str + pair[1].str
        if merged_str in self.str2token:
            new_token = self.str2token[merged_str]
            if not new_token.present:
                new_token.restore()
                logger.info(f'Restored previously removed token {new_token.str}.')
            else:
                logger.info(f'Additional merges for {new_token.str}.')
        else:
            new_token = Token(self.new_id, merged_str, 0, left=pair[0], right=pair[1])
            self.id2token[new_token.id] = new_token
            self.str2token[new_token.str] = new_token
            self.new_id += 1
        self.events.append((EventType.MERGE, pair, new_token))
        return self._merge_token_in_words(new_token, pair, pairs)

    def _dump(self, file: Union[Path, str]):
        logger.info(f'Dumping model to {file}...')
        vocab_to_embedding = dict()
        embedding_idx = 0
        for vocab_id in sorted(self.id2token.keys()):
            if self.id2token[vocab_id].present:
                vocab_to_embedding[vocab_id] = embedding_idx
                embedding_idx += 1
        with open(file, "w", encoding="utf-8") as f:
            json.dump({
                'tokens': [token.to_dict() for token in self.id2token.values()],
                'id2int': vocab_to_embedding,
                'int2id': {v: k for k, v in vocab_to_embedding.items()},
                'merges': [{
                    'id': i,
                    'pair': [token.to_dict() for token in merge[1]],
                    'new_token': merge[2].to_dict()
                } for i, merge in enumerate(self.events) if merge[0] == EventType.MERGE],
                'splits': [{
                    'id': i,
                    'token': merge[1].to_dict(),
                    'split': [token.to_dict() for token in merge[2]]
                } for i, merge in enumerate(self.events) if merge[0] == EventType.SPLIT],
            }, f, indent=4)

    def _fit_from_objects(self, words: list[Word], output_path: Union[Path, str], logging_step: int) -> Path:
        self._initialize_vocab(words)
        self._encode_words(words)
        pairs = self._initialize_pairs(words)
        merge_time = []
        while self.actual_vocab_size < self.desired_vocab_size:
            start_time = time.time()

            pair, count = pairs.most_common(1)[0]
            if count <= 0:
                logger.info(f'No more pairs to merge. Stopping with vocab size of {self.actual_vocab_size}.')
                break

            freq = self._merge_pair(pair, pairs)
            self.actual_vocab_size += 1

            merge_time.append(time.time() - start_time)
            if self.actual_vocab_size % logging_step == 0:
                logger.info(
                    f'VOCABULARY SIZE: {self.actual_vocab_size}. '
                    f'Merged {pair[0].str} + {pair[1].str} with frequency {freq}. '
                    f'Average merge time {np.mean(merge_time):.2f}s.'
                )
                merge_time = []

        self._dump(output_path)
        return Path(output_path).resolve()

    def _fit_from_counts(self, pretoken_counts: MCounter, output_path: Union[Path, str], logging_step: int) -> Path:
        return self._fit_from_objects(self._pretokens_to_objects(pretoken_counts), output_path, logging_step)

    def fit_from_file(self, input_file: Union[Path, str], model_file: Union[Path, str], logging_step: int=200) -> Path:
        return self._fit_from_counts(self._count_words_in_file(input_file), model_file, logging_step)
