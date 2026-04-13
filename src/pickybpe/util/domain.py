from __future__ import annotations

from .counters import *


WHITESPACE = '▁'


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
        self.pairs: MulCounter[Pair] = None  # Maps each adjacent pair of tokens to its corpus frequency conditioned on this word. (The amount of times it appears in this word, times self.freq.) No validation is run on these pairs. It is up to the user to decide that a pair is illegal e.g. because one of its members is special or the result of merging the pair would be too long.

    def initialize_tokens(self, str2token: dict[str, Token]) -> None:
        self.tokens = [str2token[c] for c in self.atoms]
        self._recalculate()

    @property
    def _str(self):
        return "".join(self.atoms)

    def __repr__(self) -> str:
        return f"{self._str} ({self.freq})"

    def _recalculate(self, relink_word_to_tokens: bool=True) -> None:
        self.pairs = MulCounter(skipImmediateEquals(zip(self.tokens[:-1], self.tokens[1:]))) * self.freq
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


def skipImmediateEquals(iterable: Iterable[T]) -> Iterator[T]:
    """
    Skips each element which follows an element that is (1) equal to it and (2) not itself skipped already.
    The function that makes sure that a span of tokens like x x x x x registers as 2 pairs (x,x) rather than 4.
    (Doesn't work to deduplicate None.)
    """
    prev = None
    for thing in iterable:
        if prev is not None and thing == prev:
            prev = None
            continue

        prev = thing
        yield thing
