# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from torchtune.data import truncate

from torchtune.modules.tokenizers import BaseTokenizer
from torchtune.modules.transforms import Transform

# leave space for 128 supplementary tokens
OFFSET = 3
SUPPL_TOKEN_OFFSET = OFFSET + 256
TOKEN_NUM = SUPPL_TOKEN_OFFSET + 128

class OctetTokenizer(BaseTokenizer, Transform):
    """
    A non-trainable tokenizer that simple encodes strings as UTF-8 and uses
    their octets.

    Examples:
        >>> tokenizer = OctetTokenizer()
        >>> tokenized_text = tokenizer.encode("Hello world!", add_bos=True, add_eos=True)
        >>> print(tokenized_text)
        [1, 31587, 29644, 102, 2]
    """
    pad_id = 0
    bos_id = 1
    eos_id = 2

    def __init__(self, max_seq_len: int):
        self.max_seq_len = max_seq_len

    def encode(self,
               text: str,
               add_bos: bool = True,
               add_eos: bool = True) -> List[int]:
        """
        Encode text into token IDs.

        Args:
            text: The input text to be encoded, unbatched.
            add_bos: Whether to prepend BOS to the input, defaults to True.
            add_eos: Whether to append EOS to the input, defaults to True.

        Returns:
            List[int]: The encoded token IDs.
        """
        tokens = []
        if add_bos:
            tokens.append(self.bos_id)
        tokens.extend([i + OFFSET for i in text.encode("utf-8")])
        if add_eos:
            tokens.append(self.eos_id)
        return tokens


    def decode(self, ids: 'IntTensor') -> str:
        """Decode a sequence of token IDs into a string.

        Args:
            ids: The input token IDs to be decoded.

        Returns:
            A decoded string.
        """
        ids = [id - OFFSET for id in ids if OFFSET <= id < SUPPL_TOKEN_OFFSET]
        return bytes(ids).decode("utf-8", errors="ignore")
