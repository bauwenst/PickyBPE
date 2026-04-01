from typing import Optional, Iterable, TypeVar, Union, Generic
from abc import ABC, abstractmethod

from collections import Counter
from heapdict import heapdict


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


class PairScores(ABC, Generic[T]):
    @abstractmethod
    def has(self, pair: T) -> bool:
        pass

    @abstractmethod
    def get(self, pair: T) -> float:  # Raises when the key is unknown.
        pass

    @abstractmethod
    def set(self, pair: T, value: float):
        pass

    def increment(self, pair: T, delta: float=1) -> float:  # Creates the key if it is unknown.
        try:
            value = self.get(pair) + delta
        except KeyError:
            value = delta
        self.set(pair, value)
        return value

    def decrement(self, pair: T, delta: float=1) -> float:
        return self.increment(pair, -delta)

    @abstractmethod
    def pop(self, pair: T) -> float:
        pass

    @abstractmethod
    def __iter__(self) -> Iterable[T]:
        pass


class PairScoresArgmaxable(PairScores[T]):
    """A pair collection which is also used for BPE's argmax."""

    @abstractmethod
    def get_argmax(self) -> tuple[T,float]:
        pass

    def pop_argmax(self) -> tuple[T,float]:
        pair, freq = self.get_argmax()
        self.pop(pair)
        return pair, freq


class PairCounter(PairScores[T]):
    def __init__(self):
        self._counts: MCounter[T] = MCounter()

    def has(self, pair: T) -> bool:
        return pair in self._counts

    def pop(self, pair: T) -> float:
        return self._counts.pop(pair)

    def get(self, pair: T) -> float:
        return self._counts[pair]

    def set(self, pair: T, value: float):
        self._counts[pair] = value

    def __iter__(self) -> Iterable[T]:
        return iter(self._counts)


class PairCounterArgmaxable(PairCounter[T]):
    def __init__(self):
        super().__init__()
        self._argmax: T = None  # cache to avoid double computations.

    def get_argmax(self) -> tuple[T,float]:
        if self._argmax is None:
            self._argmax = max(self._counts.keys(), key=self._counts.get)
        return self._argmax, self.get(self._argmax)

    def pop(self, pair: T) -> float:
        value = super().pop(pair)
        if self._argmax == pair:
            self._argmax = None
        return value

    def set(self, pair: T, value: float):
        super().set(pair, value)
        self._try_replace_argmax(pair)

    def increment(self, pair: T, delta: float=1) -> float:  # Slightly more efficient version than super().increment() since it sometimes skips the argmax replacement.
        try:
            value = self.get(pair) + delta
        except KeyError:
            value = delta
        super().set(pair, value)
        if delta > 0:  # <---
            self._try_replace_argmax(pair)
        return value

    def _try_replace_argmax(self, pair: T):
        if self._argmax is None:
            return  # We don't want to do any argmax-related computations unless the user asks for it. If the argmax is unknown, that means either we compute it here now and do a bunch updates on it, or we do a bunch of updates and then compute it.
        if self._argmax != pair and self.get(pair) > self.get(self._argmax):
            self._argmax = pair


class PairHeap(PairScoresArgmaxable[T]):

    def __init__(self):
        self._minheap = heapdict()  # Stores negative frequencies.

    def has(self, pair: T) -> bool:
        return pair in self._minheap

    def get_argmax(self) -> tuple[T,float]:
        pair, negfreq = self._minheap.peekitem()
        return pair, -negfreq

    def pop_argmax(self) -> tuple[T,float]:  # Slightly more efficient.
        pair, negfreq = self._minheap.popitem()
        return pair, -negfreq

    def set(self, pair: T, value: float):
        self._minheap[pair] = -value

    def get(self, pair: T) -> float:
        return -self._minheap[pair]

    def pop(self, pair: T) -> float:
        return -self._minheap.pop(pair)

    def __iter__(self) -> Iterable[T]:
        return iter(self._minheap)


class PairStatistics(ABC, Generic[T]):
    """
    Tracks both raw pair counts as well as the metric used to choose BPE merges.
    """

    @property
    @abstractmethod
    def counts(self) -> PairScores[T]:
        pass

    @abstractmethod
    def has(self, pair: T) -> bool:
        pass

    @abstractmethod
    def pop(self, pair: T) -> tuple[int,float]:
        pass

    @abstractmethod
    def recompute_objective(self, pairs: set[T]):
        pass

    @abstractmethod
    def get_argmax_objective(self) -> T:
        pass

    def pop_argmax_objective(self) -> tuple[T, int, float]:
        pair = self.get_argmax_objective()
        freq, score = self.pop(pair)
        return pair, freq, score

