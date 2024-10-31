from torchtune.models.llama3_2._component_builders import llama3_2
from torchtune.modules import TransformerDecoder

from bytellama.tokenizer import OctetTokenizer


def octet_tokenizer(*args, **kwargs) -> OctetTokenizer:
    return OctetTokenizer()


def byte_llama() -> TransformerDecoder:
    """
    Builder for creating a Llama3.2 model initialized w/ ~100M parameters

    Returns:
        TransformerDecoder: Instantiation of Llama3.2 100M model
    """
    return llama3_2(vocab_size=259,
                    num_layers=30,
                    num_heads=9,
                    num_kv_heads=3,
                    embed_dim=576,
                    max_seq_len=131072,
                    intermediate_dim=1536,
                    attn_dropout=0.0,
                    norm_eps=1e-5,
                    rope_base=10000,
                    scale_factor=32)
