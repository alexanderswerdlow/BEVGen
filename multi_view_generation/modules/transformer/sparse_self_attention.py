"""
Copyright 2020 The Microsoft DeepSpeed Team
"""

import torch.nn as nn
import torch
from torch import distributed as dist
from deepspeed.ops.sparse_attention import SparsityConfig
from einops import rearrange

class SparseSelfAttention(nn.Module):
    """Implements an efficient Sparse Self Attention of Transformer layer based on `Generative Modeling with Sparse Transformers`: https://arxiv.org/abs/1904.10509

    For more information please see, TODO DeepSpeed Sparse Transformer.

    For usage example please see, TODO DeepSpeed Sparse Transformer Tutorial.
    """
    def __init__(
            self,
            # SparsityConfig parameters needs to be set accordingly
            sparsity_config=SparsityConfig(num_heads=4),
            key_padding_mask_mode='add',
            attn_mask_mode='mul',
            max_seq_length=2048):
        """Initialize the sparse self attention layer.
        Arguments:
            sparsity_config: optional: this parameter determines sparsity pattern configuration; it is based on SparsityConfig class.
            key_padding_mask_mode: optional: a string determining if key padding mask needs to be added, `add`, or be multiplied, `mul`.
            attn_mask_mode: optional: a string determining if attention mask needs to be added, `add`, or be multiplied, `mul`.
            max_seq_length: optional: the maximum sequence length this sparse attention module will be applied to; it controls the size of the master_layout.
        """
        super().__init__()

        # sparsity information
        self.sparsity_config = sparsity_config

        # initialize sparse layout and register as buffer
        master_layout = self.sparsity_config.make_layout(max_seq_length)
        self.register_buffer("master_layout", master_layout)
        self._need_layout_synchronization = True

        # mask modes
        self.key_padding_mask_mode = key_padding_mask_mode
        self.attn_mask_mode = attn_mask_mode

    ops = dict()

    def get_layout(self, L):
        # if layout is never synchronized across GPUs, broadcast the layout from global rank 0
        if self._need_layout_synchronization and dist.is_initialized():
            dist.broadcast(self.master_layout, src=0)
            self._need_layout_synchronization = False

        if (L % self.sparsity_config.block != 0):
            raise ValueError(
                f'Sequence Length, {L}, needs to be dividable by Block size {self.sparsity_config.block}!'
            )

        num_blocks = L // self.sparsity_config.block
        return self.master_layout[..., :num_blocks, :num_blocks].cpu()  # layout needs to be a CPU tensor

    # add to cache
    def get_ops(self, H, L):
        from deepspeed.ops.sparse_attention.matmul import MatMul
        from deepspeed.ops.sparse_attention.softmax import Softmax
        if L not in SparseSelfAttention.ops:
            sparsity_layout = self.get_layout(L)
            sparse_dot_sdd_nt = MatMul(sparsity_layout,
                                       self.sparsity_config.block,
                                       'sdd',
                                       trans_a=False,
                                       trans_b=True)

            sparse_dot_dsd_nn = MatMul(sparsity_layout,
                                       self.sparsity_config.block,
                                       'dsd',
                                       trans_a=False,
                                       trans_b=False)

            sparse_softmax = Softmax(sparsity_layout, self.sparsity_config.block)

            SparseSelfAttention.ops[L] = (sparse_dot_sdd_nt,
                                          sparse_dot_dsd_nn,
                                          sparse_softmax)
        return SparseSelfAttention.ops[L]

    def transpose_key_for_scores(self, x, L):
        bsz, num_heads, seq_len, head_dim = x.size()
        if seq_len != L:
            return x.permute(0, 1, 3, 2)
        return x

    def transpose_mask_for_sparse(self, qtype, x, is_key_padding_mask=False):
        x = x.type(qtype)
        if is_key_padding_mask:
            xdim = x.dim()
            for d in range(xdim - 1, 0, -1):
                x = x.squeeze(dim=d)
            return x
        return x.squeeze()

    # forward pass
    def forward(self,
                query,
                key,
                value,
                rpe=None,
                key_padding_mask=None,
                attn_mask=None,
                add_mask=None
                ):
        """Applies forward phase of sparse self attention

        Arguments:
            query: required: query tensor
            key: required: key tensor
            value: required: value tensor
            rpe: optional: a tensor same dimension as x that is used as relative position embedding
            key_padding_mask: optional: a mask tensor of size (BatchSize X SequenceLength)
            attn_mask: optional: a mask tensor of size (SequenceLength X SequenceLength); currently only 2D is supported
            key_padding_mask_mode: optional: a boolean determining if key_padding_mask needs to be added or multiplied
            attn_mask_mode: optional: a boolean determining if attn_mask needs to be added or multiplied

        Return:
             attn_output: a dense tensor containing attention context
        """
        assert query.dtype == torch.half, "sparse attention only supports training in fp16 currently, please file a github issue if you need fp32 support"
        bsz, num_heads, tgt_len, head_dim = query.size()

        # transpose back key if it is already transposed
        key = self.transpose_key_for_scores(key, tgt_len)

        # check that operation is supported
        if query.shape != key.shape or key.shape != value.shape:
            raise NotImplementedError('only self-attention is supported for now')

        # squeeze key_padding_mask if it is given
        if key_padding_mask is not None:
            key_padding_mask = self.transpose_mask_for_sparse(query.dtype,
                                                              key_padding_mask,
                                                              is_key_padding_mask=True)

        # squeeze attn_mask if it is given
        if attn_mask is not None:
            attn_mask = self.transpose_mask_for_sparse(query.dtype, attn_mask)

        # cache look-up table computations etc
        sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax = self.get_ops(num_heads, tgt_len)

        scaling = float(head_dim)**-0.5

        # attention scores
        attn_output_weights = sparse_dot_sdd_nt(query, key)

        if add_mask is not None:
            attn_add_mask = torch.zeros_like(attn_output_weights, dtype=add_mask.dtype)
            lut_cache_key = next(iter(sparse_dot_sdd_nt.lut_cache))
            seg_len = len(sparse_dot_sdd_nt.lut_cache[lut_cache_key][0])
            core = torch.cat([sparse_dot_sdd_nt.lut_cache[lut_cache_key][0][i] for i in range(seg_len)], 0).to(torch.long)
            add_mask = rearrange(add_mask, 'b (h block_size_h) (w block_size_w) -> b h w block_size_h block_size_w', h=tgt_len // self.sparsity_config.block, w=tgt_len // self.sparsity_config.block, block_size_h=self.sparsity_config.block, block_size_w=self.sparsity_config.block)

            attn_add_mask[:, core[:, -1]] = add_mask[:,  core[:, 1], core[:, 2]]
            attn_output_weights = attn_output_weights + attn_add_mask

        # (flip: idx, columns, rows, head)
        attn_output_weights = sparse_softmax(
            attn_output_weights,
            scale=scaling,
            rpe=rpe,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            key_padding_mask_mode=self.key_padding_mask_mode,
            attn_mask_mode=self.attn_mask_mode)

        # outputs
        attn_output = sparse_dot_dsd_nn(attn_output_weights, value)
        return attn_output
