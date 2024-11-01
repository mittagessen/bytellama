Description
===========

ByteLlama is a tiny Llama 3.2 model (~101M parameters) using octet
tokenization. Its primary purpose is to server as a modernized alternative to
`ByT5 <http://arxiv.org/abs/2105.13626>`_, in particular the small version, in my
vision experiments but it should work for any application were a small LM
witohut the drawbacks of tokenization is necessary.

ByteLlama's hyperparameters were shamelessly pilfered from
`SmolLM-135M <https://huggingface.co/HuggingFaceTB/SmolLM-135M>`_. The difference
in parameter count (~34M) comes from the difference in tokenization.

This repository contains configuration and tokenization code to train a
ByteLlama using the torchtune framework.

Want to try it out?
===================

First install the package from the repository:

::

        $ pip install .

Then run the script creating the randomly initialized weights:

::

        $ bytellama ~/bitey_llamas/model.pt

To start the training on 4 GPUs on a single node with torchtune (includes
automatic download of the dataset) from the root directory of the git repository:

::

         tune run --nproc_per_node 4 full_finetune_distributed \
                --config configs/bytellama.yaml \
                checkpointer.checkpoint_dir=~/bitey_llamas \
                checkpointer.output_dir=~/bitey_llamas 


ByteLlama is *tiny*. It can fit a batch size of 32 onto a A40 GPU when
using bf16 precision. To adjust to your actual available memory.
