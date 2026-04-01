from __future__ import annotations

from typing import Optional, Iterable, TypeVar, Union
from abc import ABC, abstractmethod
from pathlib import Path

from collections import Counter
from heapdict import heapdict

WHITESPACE = '▁'
PAD = '<PAD>'
UNK = '<UNK>'
BOS = '<BOS>'
EOS = '<EOS>'

PathLike = Union[str, Path]

T = TypeVar("T")
class MCounter(Counter[T]):
    """This is a slight extension of the ``Collections.Counter`` class
    to also allow multiplication with integers.
    https://stackoverflow.com/a/74830621"""

    def __mul__(self, other):
        if not isinstance(other, int):
            raise TypeError("Non-int factor")
        return MCounter({k: other * v for k, v in self.items()})

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        return MCounter(super().__add__(other))


class Token:

    def __init__(
        self,
        id: int,
        str: str,
        freq: int = 0,
        special: bool = False,
        present: bool = True,
        left: Optional[Token] = None,
        right: Optional[Token] = None,
        split: Optional[list[Token]] = None
    ):
        self.id = id
        self.str = str
        self.freq = freq
        self.special = special
        self.present = present
        self.atomic = len(str) == 1 or special
        self.words = set()
        self.left = left
        self.right = right
        self.split = split

    def __repr__(self):
        return f'{self.str} ({self.freq})'

    def walk(self) -> list[Token]:
        if self.atomic or self.present:
            return [self]
        return self.left.walk() + self.right.walk()

    def remove(self) -> None:
        if self.atomic:
            raise ValueError(f'Cannot remove an atomic token {self.str}.')
        self.present = False
        self.freq = 0
        self.words = set()

    def restore(self) -> None:
        if self.present:
            raise ValueError(f'Cannot revoke already present token {self.str}.')
        self.present = True

    def split_if_possible(self) -> Optional[list[Token]]:
        if self.atomic:
            return None
        self.present = False
        return self.walk()

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'str': self.str,
            'freq': self.freq,
            'special': self.special,
            'present': self.present,
            'left': self.left.id if self.left is not None else None,
            'right': self.right.id if self.right is not None else None,
            'split': [t.id for t in self.walk()]
        }


Pair = tuple[Token, Token]


class Word:

    def __init__(self, id: int, atoms: Iterable[str], freq: int = 0):
        self.id = id
        self.atoms = tuple(atoms)
        self.freq = freq
        self.tokens: list[Token] = None
        self.pairs: MCounter[Pair] = None  # Maps each adjacent pair of tokens to its corpus frequency conditioned on this word. (The amount of times it appears in this word, times self.freq.) No validation is run on these pairs. It is up to the user to decide that a pair is illegal e.g. because one of its members is special or the result of merging the pair would be too long.

    def initialize_tokens(self, str2token: dict[str, Token]) -> None:
        self.tokens = [str2token[c] for c in self.atoms]
        self._recalculate()

    @property
    def _str(self):
        return "".join(self.atoms)

    def __repr__(self) -> str:
        return f"{self._str} ({self.freq})"

    def _recalculate(self, relink_word_to_tokens: bool=True) -> None:
        self.pairs = MCounter(zip(self.tokens[:-1], self.tokens[1:])) * self.freq
        if relink_word_to_tokens:
            for token in self.tokens:
                token.words.add(self)

    def merge_pair(self, pair: Pair, new_token: Token, relink_word_to_tokens: bool=True) -> int:
        new_tokens = []
        i = 0
        while i < len(self.tokens):
            if i < len(self.tokens) - 1 and (self.tokens[i], self.tokens[i+1]) == pair:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(self.tokens[i])
                i += 1
        n_tokens_created_per_word = len(self.tokens) - len(new_tokens)
        if relink_word_to_tokens:
            pair[0].words.discard(self)
            pair[1].words.discard(self)
        self.tokens = new_tokens
        self._recalculate(relink_word_to_tokens=relink_word_to_tokens)
        return n_tokens_created_per_word * self.freq

    def split_token(self, token: Token, subtokens: list[Token], relink_word_to_tokens: bool=True):
        new_tokens = []
        for t in self.tokens:
            if t == token:
                new_tokens.extend(subtokens)
            else:
                new_tokens.append(t)
        self.tokens = new_tokens
        self._recalculate(relink_word_to_tokens=relink_word_to_tokens)


class PairScores(ABC):
    @abstractmethod
    def has(self, pair: Pair) -> bool:
        pass

    @abstractmethod
    def get(self, pair: Pair) -> float:  # Raises when the key is unknown.
        pass

    @abstractmethod
    def set(self, pair: Pair, value: float):
        pass

    def increment(self, pair: Pair, delta: float=1) -> float:  # Creates the key if it is unknown.
        try:
            value = self.get(pair) + delta
        except KeyError:
            value = delta
        self.set(pair, value)
        return value

    def decrement(self, pair: Pair, delta: float=1) -> float:
        return self.increment(pair, -delta)

    @abstractmethod
    def pop(self, pair: Pair) -> float:
        pass

    @abstractmethod
    def __iter__(self) -> Iterable[Pair]:
        pass


class PairScoresArgmaxable(PairScores):
    """A pair collection which is also used for BPE's argmax."""

    @abstractmethod
    def get_argmax(self) -> tuple[Pair,float]:
        pass

    def pop_argmax(self) -> tuple[Pair,float]:
        pair, freq = self.get_argmax()
        self.pop(pair)
        return pair, freq


class PairCounter(PairScores):
    def __init__(self):
        self._counts: MCounter[Pair] = MCounter()

    def has(self, pair: Pair) -> bool:
        return pair in self._counts

    def pop(self, pair: Pair) -> float:
        return self._counts.pop(pair)

    def get(self, pair: Pair) -> float:
        return self._counts[pair]

    def set(self, pair: Pair, value: float):
        self._counts[pair] = value

    def __iter__(self) -> Iterable[Pair]:
        return iter(self._counts)


class PairCounterArgmaxable(PairCounter):
    def __init__(self):
        super().__init__()
        self._argmax: Pair = None  # cache to avoid double computations.

    def get_argmax(self) -> tuple[Pair,float]:
        if self._argmax is None:
            self._argmax = max(self._counts.keys(), key=self._counts.get)
        return self._argmax, self.get(self._argmax)

    def pop(self, pair: Pair) -> float:
        value = super().pop(pair)
        if self._argmax == pair:
            self._argmax = None
        return value

    def set(self, pair: Pair, value: float):
        super().set(pair, value)
        self._try_replace_argmax(pair)

    def increment(self, pair: Pair, delta: float=1) -> float:  # Slightly more efficient version than super().increment() since it sometimes skips the argmax replacement.
        try:
            value = self.get(pair) + delta
        except KeyError:
            value = delta
        super().set(pair, value)
        if delta > 0:  # <---
            self._try_replace_argmax(pair)
        return value

    def _try_replace_argmax(self, pair: Pair):
        if self._argmax is None:
            return  # We don't want to do any argmax-related computations unless the user asks for it. If the argmax is unknown, that means either we compute it here now and do a bunch updates on it, or we do a bunch of updates and then compute it.
        if self._argmax != pair and self.get(pair) > self.get(self._argmax):
            self._argmax = pair


class PairHeap(PairScoresArgmaxable):

    def __init__(self):
        self._minheap = heapdict()  # Stores negative frequencies.

    def has(self, pair: Pair) -> bool:
        return pair in self._minheap

    def get_argmax(self) -> tuple[Pair,float]:
        pair, negfreq = self._minheap.peekitem()
        return pair, -negfreq

    def pop_argmax(self) -> tuple[Pair,float]:  # Slightly more efficient.
        pair, negfreq = self._minheap.popitem()
        return pair, -negfreq

    def set(self, pair: Pair, value: float):
        self._minheap[pair] = -value

    def get(self, pair: Pair) -> float:
        return -self._minheap[pair]

    def pop(self, pair: Pair) -> float:
        return -self._minheap.pop(pair)

    def __iter__(self) -> Iterable[Pair]:
        return iter(self._minheap)
