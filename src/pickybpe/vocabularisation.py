from __future__ import annotations

from typing import Union, Iterable, TypeVar
from enum import Enum
from pathlib import Path

from collections import defaultdict, Counter
import numpy as np
import time
import json
import logging

logger = logging.getLogger(__name__)

from .utils import MCounter, WHITESPACE, PAD, UNK, BOS, EOS, Token, Word, PathLike, PairCounts, PairHeap, PairMCounter

T = TypeVar("T")
def modlog(iterable: Iterable[T], step: int, units: str="elements", message: str="\tProcessed") -> Iterable[T]:
    """
    Log a message each time after iterating over a fixed amount of elements in an iterable.
    """
    for i, thing in enumerate(iterable):
        yield thing
        if i > 0 and i % step == 0:
            logger.info(f"{message} {i:,} {units}.")


class EventType(Enum):
    MERGE = 0
    SPLIT = 1


class BPETrainer:

    def __init__(
        self,
        vocab_size: int,
        max_type_length: int = 64,
        character_coverage: float = 0.9999,
        ensured_vocabulary: Iterable[str] = None,

        include_specials: bool = True,
        pad_id: int = 0,
        unk_id: int = 1,
        bos_id: int = 2,
        eos_id: int = 3,
    ):
        self.coverage: float = character_coverage
        self.max_type_length = max_type_length
        self._ensured_vocabulary = set(ensured_vocabulary or [])

        if include_specials:
            self.desired_vocab_size = vocab_size
            self.unk_token = Token(unk_id, UNK, 0, special=True)
            self.pad_token = Token(pad_id, PAD, 0, special=True)
            self.bos_token = Token(bos_id, BOS, 0, special=True)
            self.eos_token = Token(eos_id, EOS, 0, special=True)
            specials = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        else:
            self.desired_vocab_size = vocab_size + 1  # User said they did not want to include specials. We include a special, so we pay for it.
            self.unk_token = Token(0, "[UNK]", 0, special=True)  # You need at least a default UNK due to character coverage.
            specials = [self.unk_token]
        assert self.unk_token

        max_id = max((token.id for token in specials), default=-1)

        self.str2token = {token.str: token for token in specials}
        self.str2token = defaultdict(lambda: self.unk_token, self.str2token)
        self.actual_vocab_size = len(specials)
        self.new_id            = max_id + 1

        self.events: list[Union[tuple[EventType,Token,list[Token]],tuple[EventType,list[Token],Token]]] = []

    def _count_words_in_file(self, file: str) -> Counter[str]:
        counter = Counter()

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

    def _pretokens_to_objects(self, pretoken_counts: Counter[str]) -> list[Word]:
        return [Word(i, self._string_to_atoms(pretoken), freq) for i, (pretoken, freq) in enumerate(pretoken_counts.items())]

    def _count_atoms(self, words: list[Word]) -> Counter[str]:
        counter = MCounter()
        for word in modlog(words, 500_000, "words"):
            counter.update(MCounter(word.atoms) * word.freq)
        return counter

    def _filter_atoms(self, characters: Counter[str]) -> Counter[str]:
        if self.coverage < 1:
            corpus_size = sum(characters.values())
            freq_to_remove = corpus_size - round(self.coverage * corpus_size)
            if freq_to_remove > 0:
                sorted_counter = characters.most_common()[::-1]  # Sorted low to high.
                cum_sum = np.cumsum([freq for _, freq in sorted_counter])
                num_to_remove = np.searchsorted(cum_sum, freq_to_remove)
                if num_to_remove > 0:
                    characters = characters.copy()
                    for c, _ in sorted_counter[:num_to_remove]:
                        characters.pop(c)

                logger.info(f'Replaced {num_to_remove} rare characters with UNK.')
        return characters

    def _ensure_atoms(self, characters: Counter[str]) -> Counter[str]:
        characters = characters.copy()
        for atom in self._ensured_vocabulary:
            characters[atom] += 0
        return characters

    def _initialize_vocab(self, words: list[Word]):
        logger.info('Initializing the vocabulary...')
        all_characters: Counter[str]      = self._count_atoms(words)
        filtered_characters: Counter[str] = self._ensure_atoms(self._filter_atoms(all_characters))
        for i, character in enumerate(sorted(filtered_characters)):
            token = Token(self.new_id + i, character, filtered_characters[character])
            self.str2token[token.str] = token

        self.new_id            += len(filtered_characters)
        self.actual_vocab_size += len(filtered_characters)
        logger.info(f'Initialized vocabulary with {len(filtered_characters)} unique characters.')
        logger.info(f"\tIncluded: {''.join(sorted(set(filtered_characters.keys())))}")
        logger.info(f"\tExcluded: {''.join(sorted(set(all_characters.keys()) - set(filtered_characters.keys())))}")

    def _validate_pair(self, pair: Iterable[Token]) -> bool:
        return not any(token.special for token in pair) and sum(len(token.str) for token in pair) <= self.max_type_length

    def _encode_words(self, words: list[Word]):
        logger.info("Encoding words...")
        for word in modlog(words, 500_000, "words"):
            word.initialize_tokens(self.str2token)  # Stores the resulting tokenisation in-place, and links each token to the words that contain it.

    def _initialize_pairs(self, words: list[Word]) -> PairCounts:
        pairs = PairHeap()        # Each merge constant-time in |V|, so vocabularisation is O(|V|).
        # pairs = PairMCounter()  # Each merge linear-time in |V|, so vocabularisation is O(|V|²).
        logger.info("Initializing token pairs...")
        for word in modlog(words, 500_000, "words"):
            for pair, freq_in_word_times_freq_of_word in word.pairs.items():
                pairs.increment(pair, freq_in_word_times_freq_of_word)

        to_remove = set()
        for pair in pairs:
            if not self._validate_pair(pair):
                to_remove.add(pair)
        for pair in to_remove:
            pairs.pop(pair)
        return pairs

    @staticmethod
    def _update_pairs_on_merge(new_token: Token, pair: tuple[Token, Token], pairs_formed_with_new_token: Counter, pairs: PairCounts):
        # If you have a context (a,b,c,d) and (b,c) is the merge, that means two things:
        #   1. You ADD new pairs (a,bc) and (bc,d).
        #   2. You REMOVE old pairs (a,b) and (c,d), not just (b,c).

        # Add pairs
        for p, freq in pairs_formed_with_new_token.items():
            pairs.increment(p, freq)

        # Remove pairs
        for p, freq in pairs_formed_with_new_token.items():
            if new_token not in p:
                raise ValueError(f'Pair {p} does not contain the new token {new_token}.')

            if new_token is p[0]:  # Case (bc,d) or (bc,bc)
                if new_token is p[1]:  # Case (bc,bc) which came from (b,c,b,c) so you lose the (c,b).
                    pair_to_update = (pair[1], pair[0])
                else:  # Case (bc,d) which came from (b,c,d) so you lose (c,d).
                    pair_to_update = (pair[1], p[1])
            else:  # Case (a,bc) which came from (a,b,c) so you lose (a,b).
                pair_to_update = (p[0], pair[0])

            if pairs.has(pair_to_update):
                new_freq = pairs.decrement(pair_to_update, freq)
                if new_freq <= 0:
                    pairs.pop(pair_to_update)

    def _scrutinize_parent_after_merge(self, parent: Token, child: Token, pair_frequency: int, pairs: PairCounts):
        pass

    def _merge_token_in_words(self, new_token: Token, pair_to_merge: tuple[Token, Token], pairs: PairCounts) -> int:
        # Update the words where the pair appears, and collect new pairs formed.
        actual_freq = 0
        pairs_for_update = Counter()
        for word in pair_to_merge[0].words & pair_to_merge[1].words:
            if pair_to_merge in word.pairs:
                word.pairs.pop(pair_to_merge)
                actual_freq += word.merge_pair(pair_to_merge, new_token)
                pairs_for_update.update({p: f for p, f in word.pairs.items() if self._validate_pair(p) and new_token in p})

        # Add the new pairs formed and subtract from merges that are no longer possible because of them.
        self._update_pairs_on_merge(new_token, pair_to_merge, pairs_for_update, pairs)
        new_token.freq += actual_freq

        # Update the tokens being merged, and possibly PickyBPE them.
        for token in set(pair_to_merge):
            if not token.present:
                raise ValueError(f'Token {token} is not present in the vocabulary.')
            token.freq -= actual_freq * pair_to_merge.count(token)

            # PickyBPE/ScaffoldBPE
            self._scrutinize_parent_after_merge(token, new_token, actual_freq, pairs)

        return actual_freq

    def _merge_pair(self, pair: tuple[Token, Token], pairs: PairCounts) -> int:
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
            self.str2token[new_token.str] = new_token
            self.new_id += 1
        self.events.append((EventType.MERGE, pair, new_token))
        return self._merge_token_in_words(new_token, pair, pairs)

    def _dump(self, path: PathLike):
        if path.is_dir():
            path = path / "tokenizer.json"
        logger.info(f'Dumping model to {path.as_posix()}...')

        sorted_tokens = sorted(self.str2token.values(), key=lambda token: token.id)
        vocab_to_embedding = {token.id: idx for idx, token in enumerate(filter(lambda token: token.present, sorted_tokens))}

        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                'tokens': [token.to_dict() for token in sorted_tokens],
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
            }, f, indent=4, ensure_ascii=False)

    def _fit_from_objects(self, words: list[Word], output_path: Union[Path, str], logging_step: int) -> Path:
        self._initialize_vocab(words)
        self._encode_words(words)
        pairs = self._initialize_pairs(words)
        merge_times = []
        while self.actual_vocab_size < self.desired_vocab_size:
            start_time = time.perf_counter_ns()

            pair, count = pairs.pop_argmax()
            if count <= 0:
                logger.info(f'No more pairs to merge. Stopping with vocab size of {self.actual_vocab_size}.')
                break

            freq = self._merge_pair(pair, pairs)
            self.actual_vocab_size += 1

            merge_times.append(time.perf_counter_ns() - start_time)
            if self.actual_vocab_size % logging_step == 0:
                logger.info(
                    f'|V| = {self.actual_vocab_size}. '
                    f'Last {logging_step} merges averaged {sum(merge_times)/(len(merge_times)*1_000_000):.2f}ms. '
                    f'Just merged {pair[0].str} + {pair[1].str} with frequency {freq}.'
                )
                merge_times = []

        self._dump(output_path)
        return Path(output_path).resolve()

    def _fit_from_counts(self, pretoken_counts: Counter[str], output_path: Union[Path, str], logging_step: int) -> Path:
        return self._fit_from_objects(self._pretokens_to_objects(pretoken_counts), output_path, logging_step)

    def fit_from_file(self, input_file: Union[Path, str], model_file: Union[Path, str], logging_step: int=200) -> Path:
        return self._fit_from_counts(self._count_words_in_file(input_file), model_file, logging_step)


class PickyBPETrainer(BPETrainer):

    def __init__(
        self,
        vocab_size: int,
        picky_threshold: float = 0.9,
        max_type_length: int = 64,
        character_coverage: float = 0.9999,
        ensured_vocabulary: Iterable[str] = None,

        include_specials: bool = True,
        pad_id: int = 0,
        unk_id: int = 1,
        bos_id: int = 2,
        eos_id: int = 3,
    ):
        super().__init__(
            vocab_size=vocab_size,
            max_type_length=max_type_length,
            character_coverage=character_coverage,
            ensured_vocabulary=ensured_vocabulary,
            include_specials=include_specials,

            pad_id=pad_id,
            unk_id=unk_id,
            bos_id=bos_id,
            eos_id=eos_id
        )
        self._threshold = picky_threshold

    @staticmethod
    def _update_pairs_on_remove(token: Token, split: list[Token], pairs_for_update: MCounter, pairs: PairCounts):
        for pair, freq in pairs_for_update.items():
            if token is pair[0]:
                if token is pair[1]:
                    pair_to_update = (split[-1], split[0])
                else:
                    pair_to_update = (split[-1], pair[1])
            else:
                pair_to_update = (pair[0], split[0])
            pairs.increment(pair_to_update, freq)
            pairs.pop(pair)

    def _remove_if_possible(self, token: Token, merged_freq: int, pairs: PairCounts) -> bool:
        if merged_freq / (token.freq + merged_freq) > self._threshold:
            split = token.split_if_possible()
            if split is not None:
                self.actual_vocab_size -= 1
                for t in split:
                    t.freq += token.freq
                for pair in zip(split[:-1], split[1:]):
                    pairs.increment(pair, token.freq)
                pairs_for_update = MCounter()
                for word in token.words:
                    if token not in word.tokens:
                        raise ValueError(f'Token {token} not found in the token list {word.tokens} of word {word}.')
                    pairs_for_update.update({pair: freq for pair, freq in word.pairs.items() if self._validate_pair(pair) and token in pair})
                    word.split_token(token, split)
                self._update_pairs_on_remove(token, split, pairs_for_update, pairs)
                token.remove()
                return True
        return False

    def _scrutinize_parent_after_merge(self, parent: Token, child: Token, pair_frequency: int, pairs: PairCounts):
        remaining_token_freq = parent.freq
        removed = self._remove_if_possible(parent, child.freq, pairs)
        if removed:  # Also means parent.freq == 0 due to the previous line.
            logger.info(f'Removed token {parent.str} with frequency {remaining_token_freq} after merging into {child.str} with frequency {child.freq} (merge frequency {pair_frequency}).')
            self.events.append((EventType.SPLIT, parent, parent.walk()))
