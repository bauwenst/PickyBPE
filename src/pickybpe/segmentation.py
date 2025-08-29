from typing import Union, Any
from pathlib import Path

import json
import time
import numpy as np
from collections import defaultdict
from functools import lru_cache
import logging
logger = logging.getLogger(__name__)

from .utils import WHITESPACE, UNK, Token, Word

PathLike = Union[str, Path]


class PickyBPECore:

    def __init__(
        self,
        id2token: dict[int, Token],
        str2token: dict[str, Token],
        id2int: dict[str, int],
        int2id: dict[int, str],

        merge_map: dict[tuple[Token, Token], np.ndarray],
        split_map: dict[Token, np.ndarray],
        splits: dict[int, list[Token]],
        events: list[dict[str, Any]]
    ):
        self.id2token: dict[int,Token]  = id2token
        self.str2token: dict[str,Token] = str2token
        self.id2int: dict[str,int]      = id2int
        self.int2id: dict[int,str]      = int2id

        self.merge_map: dict[tuple[Token,Token],np.ndarray] = merge_map
        self.split_map: dict[Token,np.ndarray] = split_map
        self.splits: dict[int,list[Token]]     = splits
        self.events: list[dict[str,Any]]       = events

    @classmethod
    def from_pretrained(self, pickybpe_model_path: PathLike) -> "PickyBPECore":
        with open(pickybpe_model_path, "r", encoding="utf-8") as f:
            serialised = json.load(f)

        id2token  = dict()
        str2token = dict()
        for token_dict in sorted(serialised['tokens'], key=lambda x: x['id']):
            token = Token(
                token_dict['id'],
                token_dict['str'],
                token_dict['freq'],
                token_dict['special'],
                token_dict['present'],
                id2token[token_dict['left']]  if token_dict['left']  is not None else None,
                id2token[token_dict['right']] if token_dict['right'] is not None else None,
                [id2token[i] for i in token_dict['split']] if len(token_dict['split']) > 1 else None
            )
            id2token[token.id]   = token
            str2token[token.str] = token

        assert UNK in str2token
        str2token = defaultdict(lambda: str2token[UNK], str2token)

        merge_map = defaultdict(list)
        for merge in serialised['merges']:
            merge_map[(str2token[merge['pair'][0]['str']], str2token[merge['pair'][1]['str']])].append(merge['id'])
        merge_map_numpy = dict()
        for key,value in merge_map.items():
            merge_map_numpy[key] = np.array(value)

        splits = dict()
        split_map = defaultdict(list)
        for split in serialised['splits']:
            split_map[str2token[split['token']['str']]].append(split['id'])
            splits[split['id']] = [str2token[token['str']] for token in split['split']]
        split_map_numpy = dict()
        for key,value in split_map.items():
            split_map_numpy[key] = np.array(value)

        return PickyBPECore(
            id2token=id2token,
            str2token=str2token,
            id2int=serialised['id2int'],
            int2id=serialised['int2id'],

            merge_map=merge_map_numpy,
            split_map=split_map_numpy,
            splits=splits,
            events=sorted(serialised['merges'] + serialised['splits'], key=lambda x: x['id'])
        )

    @lru_cache(maxsize=None)
    def _encode_word_by_event_sequence(self, word: str) -> list[Token]:
        if word in self.str2token and self.str2token[word].present:
            return [self.str2token[word]]

        word = Word(0, word)
        word.encode(self.str2token)
        for event in self.events:
            pairs = word.pairs
            if 'pair' in event:
                pair = (self.str2token[event['pair'][0]['str']], self.str2token[event['pair'][1]['str']])
                if pair in pairs:
                    word.merge_pair(pair, self.str2token[event['new_token']['str']])
            else:
                token = self.str2token[event['token']['str']]
                if token in word.tokens:
                    word.split_token(token, [self.str2token[t['str']] for t in event['split']])

        return word.tokens

    @lru_cache(maxsize=None)
    def _encode_word_by_events(self, word: str) -> list[Token]:
        if word in self.str2token and self.str2token[word].present:
            return [self.str2token[word]]

        previous_event = -1
        word = Word(0, word)
        word.encode(self.str2token)
        while True:
            pairs = [pair for pair in word.pairs if pair in self.merge_map]
            pairs = [(pair, self.merge_map[pair][np.searchsorted(self.merge_map[pair], previous_event)]) for pair in pairs
                     if np.any(self.merge_map[pair] >= previous_event)]
            removals = [token for token in word.tokens if token in self.split_map]
            removals = [(token, self.split_map[token][np.searchsorted(self.split_map[token], previous_event)])
                        for token in removals if np.any(self.split_map[token] >= previous_event)]
            if not pairs and not removals:
                break

            pair_to_merge, token_to_remove = None, None
            merge_event_id, split_event_id = None, None
            if pairs:
                pair_to_merge, merge_event_id = min(pairs, key=lambda p: p[1])
            if removals:
                token_to_remove, split_event_id = min(removals, key=lambda t: t[1])
            if merge_event_id is None and split_event_id is None:
                break

            if token_to_remove is None or (pair_to_merge is not None and merge_event_id < split_event_id):
                word.merge_pair(pair_to_merge, self.str2token[pair_to_merge[0].str + pair_to_merge[1].str], update_tokens=False)
                previous_event = merge_event_id
            else:
                word.split_token(token_to_remove, self.splits[split_event_id], update_tokens=False)
                previous_event = split_event_id

        return word.tokens

    def encode_file(self, input_file: str, output_file: str, return_type: str='str'):
        start_time = time.time()
        result = []
        with open(input_file, 'r') as file:
            logger.info('Encoding text...')
            for i, line in enumerate(file):
                words = line.strip().split()
                tokens = [token for word in words for token in self._encode_word_by_events(WHITESPACE + word)]
                if return_type == 'str':
                    result.append(' '.join([token.str for token in tokens]))
                elif return_type == 'int':
                    result.append(' '.join([str(self.id2int[str(token.id)]) for token in tokens]))
                else:
                    raise NotImplementedError(f'Unknown return type: {return_type}. Available options: str, int.')
                if i > 0 and i % 100000 == 0:
                    logger.info(f'Encoded {i} lines. Elapsed time: {time.time() - start_time:.2f} seconds.')

        logger.info(f'Encoded text in {time.time() - start_time:.2f} seconds.')
        start_time = time.time()
        with open(output_file, 'w') as file:
            logger.info('Writing encoded text...')
            file.write('\n'.join(result))
        logger.info(f'Wrote encoded text in {time.time() - start_time:.2f} seconds.')

    def decode(self, text: str, input_type: str = 'str') -> str:
        sentences = [sentence.strip().split(' ') for sentence in text.strip().split('\n')]
        if input_type == 'int':
            sentences = [[self.id2token[self.int2id[token]].str for token in sentence] for sentence in sentences]
        elif input_type != 'str':
            raise NotImplementedError(f'Unknown input type: {input_type}. Available options: str, int.')
        return '\n'.join([''.join(sentence).replace(WHITESPACE, ' ').strip() for sentence in sentences])
