from torchtune.models.llama3_2._component_builders import llama3_2
from torchtune.modules import TransformerDecoder

from bytellama.tokenizer import OctetTokenizer, TOKEN_NUM


def octet_tokenizer(*args, **kwargs) -> OctetTokenizer:
    return OctetTokenizer(*args, **kwargs)


def byte_llama() -> TransformerDecoder:
    """
    Builder for creating a Llama3.2 model initialized w/ ~10M parameters

    Returns:
        TransformerDecoder: Instantiation of Llama3.2 10M model
    """
    return llama3_2(vocab_size=TOKEN_NUM,
                    num_layers=12,
                    num_heads=8,
                    num_kv_heads=2,
                    embed_dim=192,
                    max_seq_len=131072,
                    intermediate_dim=1024,
                    attn_dropout=0.0,
                    norm_eps=1e-5,
                    rope_base=10000,
                    scale_factor=32)
