from __future__ import annotations

from typing import Literal, TypeVar, Optional
from pathlib import Path
from enum import Enum
from dataclasses import dataclass

from collections import defaultdict
import time
import json
import logging

logger = logging.getLogger(__name__)

from .util.domain import *
from .util.counters import *


T = TypeVar("T")
def modlog(iterable: Iterable[T], step: int, units: str="elements", message: str="\tProcessed") -> Iterable[T]:
    """
    Log a message each time after iterating over a fixed amount of elements in an iterable.
    """
    for i, thing in enumerate(iterable):
        yield thing
        if i > 0 and i % step == 0:
            logger.info(f"{message} {i:,} {units}.")


class MovingAverage:

    def __init__(self, width: int, stride: int):
        self._width  = width
        self._stride = stride
        self._samples = []

    def reset(self):
        self._samples = []

    def ready(self) -> bool:
        return len(self._samples) == self._width

    def add(self, x: float):
        """Shifts (and thus maybe invalidates) the window in case it is already full."""
        if self.ready():
            self._samples = self._samples[self._stride:]
        self._samples.append(x)

    def compute(self) -> float:
        return sum(self._samples) / len(self._samples)


class EventType(Enum):
    MERGE = 0
    SPLIT = 1


@dataclass
class BPETrainerState:
    str2token: dict[str,Token]
    actual_vocab_size: int
    corpus_token_count: int
    new_id: int
    events: list[Union[tuple[Literal[EventType.SPLIT], Token, Iterable[Token]], tuple[Literal[EventType.MERGE], Iterable[Token], Token]]]
    pairs: CountingObjective


class CountingObjective(ABC):
    """
    Tracks both raw pair counts as well as the metric used to choose BPE merges.
    """

    @property
    @abstractmethod
    def counts(self) -> NumericalMapping[Pair]:
        pass

    @abstractmethod
    def has(self, pair: Pair) -> bool:
        pass

    @abstractmethod
    def pop(self, pair: Pair) -> tuple[int,float]:
        pass

    @abstractmethod
    def recompute_objective(self, pairs_with_updated_counts: Iterable[Pair], state: BPETrainerState, subtokens: Optional[Pair]):
        """
        Re-synchronise the objective knowing that the given pairs had their counts changed in the latest token merge/removal.
        (Depending on the objective, many more pairs need to have their objective updated though.)
        """
        pass

    @abstractmethod
    def get_argmax_objective(self) -> Pair:
        pass

    def pop_argmax_objective(self) -> tuple[Pair, int, float]:
        key = self.get_argmax_objective()
        freq, score = self.pop(key)
        return key, freq, score


class RawBPEStatistics(CountingObjective):
    """
    Uses the raw BPE counts as argmaxable statistic.
    """

    def __init__(self):
        self._counts = MaxHeap()  # Each merge constant-time in |V|, so vocabularisation is O(|V|).
        # self._counts = FlatCounterArgmaxable()  # Each merge results from a linear-time search in |V|, so vocabularisation is O(|V|²).

    @property
    def counts(self) -> NumericalMapping[Pair]:
        return self._counts

    def has(self, pair: Pair) -> bool:
        return self._counts.has(pair)

    def pop(self, pair: Pair) -> tuple[int,float]:
        freq = self._counts.pop(pair)
        return freq, freq


    def recompute_objective(self, pairs_with_updated_counts: Iterable[Pair], state: BPETrainerState, subtokens: Optional[Pair]):
        pass  # No need, it's already computed.

    def get_argmax_objective(self) -> Pair:
        return self._counts.get_argmax()[0]


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
        UNK = '[UNK]'
        PAD = '[PAD]'
        BOS = '[BOS]'
        EOS = '[EOS]'

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
            self.unk_token = Token(0, UNK, 0, special=True)  # You need at least a default UNK due to character coverage.
            specials = [self.unk_token]
        assert self.unk_token
        self.specials = specials

    def __repr__args__(self) -> str:
        return f"V={self.desired_vocab_size}_l={self.max_type_length}_cc={self.coverage}"

    def _initialize_state(self) -> BPETrainerState:
        return BPETrainerState(
            str2token=defaultdict(lambda: self.unk_token, {token.str: token for token in self.specials}),
            actual_vocab_size=len(self.specials),
            corpus_token_count=0,
            new_id=max((token.id for token in self.specials), default=-1) + 1,
            events=[],
            pairs=RawBPEStatistics()
        )

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
        counter = MulCounter()
        for word in modlog(words, 500_000, "words"):
            counter.update(MulCounter(word.atoms) * word.freq)
        return counter

    def _filter_atoms(self, characters: Counter[str]) -> Counter[str]:
        if self.coverage < 1:
            characters = characters.copy()
            corpus_size = sum(characters.values())
            freq_to_remove = corpus_size - round(self.coverage * corpus_size)
            n_removed   = 0
            for atom, freq in characters.most_common()[::-1]:  # Sorted low to high.
                if atom in self._ensured_vocabulary:
                    continue

                if freq_to_remove - freq < 0:  # We use an inclusive limit. So, if there are 100 characters to remove but removing the least frequent removable character would remove 500 characters, then it is not removed.
                    break
                else:
                    freq_to_remove -= freq
                    characters.pop(atom)
                    n_removed += 1

            logger.info(f'Removed {n_removed} rare characters from the alphabet.')
        return characters

    def _ensure_atoms(self, characters: Counter[str]) -> Counter[str]:
        characters = characters.copy()
        n_added = 0
        for atom in self._ensured_vocabulary:
            if atom not in characters:
                characters[atom] += 0
                n_added += 1
        if n_added > 0:
            logger.info(f'Added {n_added} forgotten characters to the alphabet.')

        return characters

    def _initialize_vocab(self, words: list[Word], state: BPETrainerState):
        logger.info('Initializing the vocabulary...')
        all_characters: Counter[str]      = self._count_atoms(words)
        filtered_characters: Counter[str] = self._ensure_atoms(self._filter_atoms(all_characters))
        for i, character in enumerate(sorted(filtered_characters)):
            token = Token(state.new_id + i, character, filtered_characters[character])
            state.str2token[token.str] = token

        state.new_id            += len(filtered_characters)
        state.actual_vocab_size += len(filtered_characters)
        logger.info(f'Initialized vocabulary with {len(filtered_characters)} unique characters.')
        logger.info(f"\tIncluded: {''.join(sorted(set(filtered_characters.keys())))}")
        logger.info(f"\tExcluded: {''.join(sorted(set(all_characters.keys()) - set(filtered_characters.keys())))}")

    def _validate_pair(self, pair: Iterable[Token]) -> bool:
        return not any(token.special for token in pair) and sum(len(token.str) for token in pair) <= self.max_type_length

    def _encode_words(self, words: list[Word], state: BPETrainerState):
        logger.info("Encoding words...")
        for word in modlog(words, 500_000, "words"):
            word.initialize_tokens(state.str2token)  # Stores the resulting tokenisation in-place, and links each token to the words that contain it.
            state.corpus_token_count += word.freq * len(word.tokens)

    def _initialize_pairs(self, words: list[Word], state: BPETrainerState):
        logger.info("Initializing token pairs...")
        for word in modlog(words, 500_000, "words"):
            for pair, freq_in_word_times_freq_of_word in word.pairs.items():
                if self._validate_pair(pair):
                    state.pairs.counts.increment(pair, freq_in_word_times_freq_of_word)
        state.pairs.recompute_objective(state.pairs.counts, state, None)

    def _update_pairs_on_merge(self, new_token: Token, pair: tuple[Token, Token], pairs_formed_with_new_token: Counter, state: BPETrainerState):
        """
        If you have a context (a,b,c,d) and (b,c) is the merge, that means two things:
          1. You ADD new pairs (a,bc) and (bc,d), if they are valid.
          2. You REMOVE old pairs (a,b) and (c,d), not just (b,c).
        """
        affected_pairs = set()

        # Add pairs
        for new_pair, freq in pairs_formed_with_new_token.items():
            if self._validate_pair(new_pair):
                state.pairs.counts.increment(new_pair, freq)
                affected_pairs.add(new_pair)
            else:
                assert not state.pairs.has(new_pair)

        # Remove pairs
        popped_pairs = set()
        for new_pair, freq in pairs_formed_with_new_token.items():
            # There are three cases:
            #   1. Both tokens in new_pair are the new token: case (bc,bc) which came from (b,c,b,c) so you lose the (c,b).
            #   2. Only left token in new_pair is the new token: case (bc,d) which came from (b,c,d) so you lose (c,d).
            #   3. Only right token in new_pair is the new token: case (a,bc) which came from (a,b,c) so you lose (a,b).
            assert new_token in new_pair, f'Pair {new_pair} does not contain the new token {new_token}.'
            old_left_token  = pair[1] if new_pair[0] is new_token else new_pair[0]
            old_right_token = pair[0] if new_pair[1] is new_token else new_pair[1]
            old_pair = (old_left_token, old_right_token)
            if state.pairs.has(old_pair):  # Since the new token is the result of a merge of two existing tokens, it is true that IF those tokens could validly be merged with their neighbours, that has already been registered in pairs. When it hasn't been, it is probably because the pair was never valid: e.g., in [a, b, UNK], if you merge to [ab, UNK], you don't lose the merge (b, UNK).
                new_freq = state.pairs.counts.decrement(old_pair, freq)
                affected_pairs.add(old_pair)
                if new_freq <= 0:
                    state.pairs.pop(old_pair)
                    popped_pairs.add(old_pair)

        state.pairs.recompute_objective(affected_pairs - popped_pairs, state, pair)

    def _scrutinize_parent_after_merge(self, parent: Token, child: Token, pair_frequency: int, state: BPETrainerState):
        pass

    def _stopping_condition(self, latest_score: float, state: BPETrainerState) -> bool:
        return state.actual_vocab_size >= self.desired_vocab_size

    def _merge_token_in_words(self, new_token: Token, pair_to_merge: tuple[Token, Token], state: BPETrainerState) -> int:  # Assumes the merge being performed is valid.
        # Update the words where the pair appears, and collect new pairs formed.
        actual_freq = 0
        newly_created_pairs = Counter()
        for word in pair_to_merge[0].words & pair_to_merge[1].words:
            if pair_to_merge in word.pairs:
                actual_freq += word.merge_pair(pair_to_merge, new_token)  # note that this may create invalid pairs with the new token (e.g. pairs xy z too long), but we'll filter those later. The frequency of xy that comes out of this is still correct of course.
                newly_created_pairs.update({p: f for p, f in word.pairs.items() if new_token in p})  # .update is accumulative; also note that we don't filter out valid pairs here yet.

        # Update frequencies, before anything else is computed:
        #   - Globally, add +1 to |V|.
        #   - Globally, take away all the merged tokens from |T|.
        #   - Add to the new token's frequency.
        #   - Take away from the old tokens' frequencies.
        state.actual_vocab_size  += 1
        state.corpus_token_count -= actual_freq*(len(pair_to_merge)-1)
        new_token.freq += actual_freq
        for token in set(pair_to_merge):
            token.freq -= actual_freq * pair_to_merge.count(token)

        # Run formulas for PickyBPE/ScaffoldBPE
        for token in set(pair_to_merge):
            assert token.present, f"Merged token {token} was already absent in the vocabulary."
            self._scrutinize_parent_after_merge(token, new_token, actual_freq, state)

        # Run formulas for the pair objective
        self._update_pairs_on_merge(new_token, pair_to_merge, newly_created_pairs, state)
        return actual_freq

    def _merge_pair(self, pair: tuple[Token, Token], state: BPETrainerState) -> int:
        assert self._validate_pair(pair)

        # Create new type
        merged_str = pair[0].str + pair[1].str
        if merged_str in state.str2token:
            new_token = state.str2token[merged_str]
            if not new_token.present:
                new_token.restore()
                logger.info(f'Restored previously removed token {new_token.str}.')
            else:
                logger.info(f'Additional merges for {new_token.str}.')
        else:
            new_token = Token(state.new_id, merged_str, 0, left=pair[0], right=pair[1])
            state.str2token[new_token.str] = new_token
            state.new_id += 1
        state.events.append((EventType.MERGE, pair, new_token))
        return self._merge_token_in_words(new_token, pair, state)

    def _fit_from_objects(self, words: list[Word], output_path: Union[Path, str], logging_step: int) -> Path:
        state = self._initialize_state()
        self._initialize_vocab(words, state)
        self._encode_words(words, state)
        self._initialize_pairs(words, state)

        latest_score = None
        log_merge_time = MovingAverage(width=logging_step, stride=logging_step)
        while latest_score is None or not self._stopping_condition(latest_score, state):
            start_time = time.perf_counter_ns()

            pair, count, latest_score = state.pairs.pop_argmax_objective()
            if count <= 0:  # TODO: I feel like this probably never happens and instead the above line will crash.
                logger.info(f'No more pairs to merge. Stopping with vocab size of {state.actual_vocab_size}.')
                break

            log_new_freq = self._merge_pair(pair, state)  # TODO: Shouldn't this equal 'count'?

            log_merge_time.add(time.perf_counter_ns() - start_time)
            if log_merge_time.ready():
                logger.info(
                    f'|V| = {state.actual_vocab_size}. '
                    f'Last {logging_step} merges averaged {log_merge_time.compute()/1_000_000:.2f}ms. '
                    f'Just merged {pair[0].str} + {pair[1].str} with frequency {log_new_freq}.'
                )

        self._dump(state, output_path)
        return Path(output_path).resolve()

    def _fit_from_counts(self, pretoken_counts: Counter[str], output_path: Union[Path, str], logging_step: int) -> Path:
        return self._fit_from_objects(self._pretokens_to_objects(pretoken_counts), output_path, logging_step)

    def fit_from_file(self, input_file: Union[Path, str], model_file: Union[Path, str], logging_step: int=200) -> Path:
        return self._fit_from_counts(self._count_words_in_file(input_file), model_file, logging_step)

    def _dump(self, state: BPETrainerState, path: Path):
        if path.is_dir():
            path = path / "tokenizer.json"
        logger.info(f'Dumping model to {path.as_posix()}...')

        sorted_tokens = sorted(state.str2token.values(), key=lambda token: token.id)
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
                } for i, merge in enumerate(state.events) if merge[0] == EventType.MERGE],
                'splits': [{
                    'id': i,
                    'token': merge[1].to_dict(),
                    'split': [token.to_dict() for token in merge[2]]
                } for i, merge in enumerate(state.events) if merge[0] == EventType.SPLIT],
            }, f, indent=4, ensure_ascii=False)


########################################################################################################################


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

    def __repr__args__(self) -> str:
        return super().__repr__args__() + f"_t={self._threshold}"

    def _update_pairs_on_remove(self, removed_token: Token, subtokens: list[Token], obsolete_pairs: Counter, state: BPETrainerState):
        """
        Note that unlike in the case of a merge, the given pairs are BEFORE the event happened, not AFTER. That means
        we first need to deduce the new pair before we can increment its frequency.
        """
        created_pairs = set()

        assert len(subtokens) >= 2
        first_subtoken = subtokens[0]
        last_subtoken  = subtokens[-1]
        for old_pair, freq in obsolete_pairs.items():
            assert removed_token in old_pair
            new_left_token  = last_subtoken  if old_pair[0] is removed_token else old_pair[0]
            new_right_token = first_subtoken if old_pair[1] is removed_token else old_pair[1]
            new_pair = (new_left_token, new_right_token)

            # Do the equivalent of what is done at the start of _update_pairs_on_merge, namely incrementing new pairs if they are valid.
            if self._validate_pair(new_pair):
                state.pairs.counts.increment(new_pair, freq)
                created_pairs.add(new_pair)
            else:
                assert not state.pairs.has(new_pair)

            # And now do what happens at the bottom of _update_pairs_on_merge, namely checking "Was the old pair registered? If yes, decrement its count now that the merge is no longer available."
            if state.pairs.has(old_pair):
                assert state.pairs.counts.get(old_pair) == freq  # We don't do a decrement and zero check, because we know for sure that all instances of removed_token were considered and thus all instances of it with the other token in this old pair. In other words: this pair cannot exist elsewhere.
                state.pairs.pop(old_pair)

        state.pairs.recompute_objective(created_pairs, state, subtokens)

    def _remove_if_possible(self, token: Token, merged_freq: int, state: BPETrainerState) -> bool:
        if merged_freq / (token.freq + merged_freq) > self._threshold:
            subtokens = token.split_if_possible()
            if subtokens is not None:
                # Token-internal statistics: The token itself disappears, the subtoken unigrams appear more, and the pair(s) formed by the (at least 2) subtokens also appears more now.
                token.remove()
                for t in subtokens:
                    t.freq += token.freq
                state.actual_vocab_size -= 1
                state.corpus_token_count += token.freq * (len(subtokens) - 1)  # Has to happen before recomputation of the objective. -1 because xyz xyz (2 tokens) turns into x y z x y z (6 tokens == 2 existing tokens + (3-1)*2 new tokens)

                for pair in zip(subtokens[:-1], subtokens[1:]):
                    if self._validate_pair(pair):
                        state.pairs.counts.increment(pair, token.freq)
                    else:
                        assert not state.pairs.has(pair)

                # Token-external statistics: Every pair that used to exist with the removed token, in every word it appeared in, is now gone, and instead you get pairs with the outermost subtokens.
                obsolete_pairs = MulCounter()
                for word in token.words:
                    assert token in word.tokens, f"Token {token} not found in the token list {word.tokens} of word {word} despite that word being known to the token!"
                    obsolete_pairs.update({pair: freq for pair, freq in word.pairs.items() if token in pair})  # Note that MCounter.update is accumulative, not substitutive.
                    word.split_token(token, subtokens)  # Update token and pair information stored by the word. Doesn't matter if this produces any pair that is invalid. (No mention of the cumulative 'pairs' counts is made here yet; increments of the internal pairs was done above and increments and decrements of the edge pairs will be done below, neither needing accessing to specific words.)

                self._update_pairs_on_remove(token, subtokens, obsolete_pairs, state)  # All the way at the end.
                return True
        return False

    def _scrutinize_parent_after_merge(self, parent: Token, child: Token, pair_frequency: int, state: BPETrainerState):
        remaining_token_freq = parent.freq
        removed = self._remove_if_possible(parent, child.freq, state)
        if removed:  # Also means parent.freq == 0 due to the previous line.
            logger.info(f'Removed token {parent.str} with frequency {remaining_token_freq} after merging into {child.str} with frequency {child.freq} (merge frequency {pair_frequency}).')
            state.events.append((EventType.SPLIT, parent, parent.walk()))
