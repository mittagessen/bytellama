# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from torchtune.modules.tokenizers._utils import BaseTokenizer

class OctetTokenizer(BaseTokenizer):
    """
    A non-trainable tokenizer that simple encodes strings as UTF-8 and uses
    their octets.

    Examples:
        >>> tokenizer = OctetTokenizer()
        >>> tokenized_text = tokenizer.encode("Hello world!", add_bos=True, add_eos=True)
        >>> print(tokenized_text)
        [1, 31587, 29644, 102, 2]
    """

    def __init__(self):
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self._offset = 3

    def encode(self,
               text: str,
               add_bos: bool = True,
               add_eos: bool = True) -> List[int]:
        """
        Encode text into token IDs.

        Args:
            text (str): The input text to be encoded, unbatched.
            add_bos (bool): Whether to prepend BOS to the input, defaults to True.
            add_eos (bool): Whether to append EOS to the input, defaults to True.

        Returns:
            List[int]: The encoded token IDs.
        """
        tokens = []
        if add_bos:
            tokens.append(self.bos_id)
        tokens.extend([i + self._offset for i in text.encode("utf-8")])
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to strings.

        Args:
            ids (List[int]): The input token IDs to be decoded.

        Returns:
            str: The decoded text.
        """
        string = bytes([x - self._offset for x in ids]).decode("utf-8", errors="ignore")
        return string
