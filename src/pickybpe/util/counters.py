from typing import Optional, Iterable, TypeVar, Union, Generic, Iterator
from abc import ABC, abstractmethod

from collections import Counter
from heapdict import heapdict


T = TypeVar("T")
class MulCounter(Counter[T]):
    """This is a slight extension of the ``Collections.Counter`` class
    to also allow multiplication with integers.
    https://stackoverflow.com/a/74830621"""

    def __mul__(self, other):
        if not isinstance(other, int):
            raise TypeError("Non-int factor")
        return MulCounter({k: other * v for k, v in self.items()})

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        return MulCounter(super().__add__(other))


class NumericalMapping(ABC, Generic[T]):
    """
    Maps things to a number.
    Basically a dict[T,float] except with extra methods for in/decrements, and without magic methods.
    """

    @abstractmethod
    def has(self, key: T) -> bool:
        pass

    @abstractmethod
    def get(self, key: T) -> float:  # Raises when the key is unknown.
        pass

    @abstractmethod
    def set(self, key: T, value: float):
        pass

    def increment(self, key: T, delta: float=1) -> float:  # Creates the key if it is unknown.
        try:
            value = self.get(key) + delta
        except KeyError:
            value = delta
        self.set(key, value)
        return value

    def decrement(self, key: T, delta: float=1) -> float:
        return self.increment(key, -delta)

    @abstractmethod
    def pop(self, key: T) -> float:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        pass


class NumericalMappingArgmaxable(NumericalMapping[T]):
    """A pair collection which is also used for BPE's argmax."""

    @abstractmethod
    def get_argmax(self) -> tuple[T,float]:
        pass

    def pop_argmax(self) -> tuple[T,float]:
        pair, freq = self.get_argmax()
        self.pop(pair)
        return pair, freq


class FlatCounter(NumericalMapping[T]):
    def __init__(self):
        self._counts: MulCounter[T] = MulCounter()

    def has(self, key: T) -> bool:
        return key in self._counts

    def pop(self, key: T) -> float:
        return self._counts.pop(key)

    def get(self, key: T) -> float:
        return self._counts[key]

    def set(self, key: T, value: float):
        self._counts[key] = value

    def __iter__(self) -> Iterable[T]:
        return iter(self._counts)


class FlatCounterArgmaxable(FlatCounter[T]):
    def __init__(self):
        super().__init__()
        self._argmax: T = None  # cache to avoid double computations.

    def get_argmax(self) -> tuple[T,float]:
        if self._argmax is None:
            self._argmax = max(self._counts.keys(), key=self._counts.get)
        return self._argmax, self.get(self._argmax)

    def pop(self, key: T) -> float:
        value = super().pop(key)
        if self._argmax == key:
            self._argmax = None
        return value

    def set(self, key: T, value: float):
        super().set(key, value)
        self._try_replace_argmax(key)

    def increment(self, key: T, delta: float=1) -> float:  # Slightly more efficient version than super().increment() since it sometimes skips the argmax replacement.
        try:
            value = self.get(key) + delta
        except KeyError:
            value = delta
        super().set(key, value)
        if delta > 0:  # <---
            self._try_replace_argmax(key)
        return value

    def _try_replace_argmax(self, key: T):
        if self._argmax is None:
            return  # We don't want to do any argmax-related computations unless the user asks for it. If the argmax is unknown, that means either we compute it here now and do a bunch updates on it, or we do a bunch of updates and then compute it.
        if self._argmax != key and self.get(key) > self.get(self._argmax):
            self._argmax = key


class MaxHeap(NumericalMappingArgmaxable[T]):

    def __init__(self):
        self._minheap = heapdict()  # Stores negative frequencies.

    def has(self, key: T) -> bool:
        return key in self._minheap

    def get_argmax(self) -> tuple[T,float]:
        key, negfreq = self._minheap.peekitem()
        return key, -negfreq

    def pop_argmax(self) -> tuple[T,float]:  # Slightly more efficient.
        pair, negfreq = self._minheap.popitem()
        return pair, -negfreq

    def set(self, key: T, value: float):
        self._minheap[key] = -value

    def get(self, key: T) -> float:
        return -self._minheap[key]

    def pop(self, key: T) -> float:
        return -self._minheap.pop(key)

    def __iter__(self) -> Iterable[T]:
        return iter(self._minheap)
