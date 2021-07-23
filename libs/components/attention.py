import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional, Tuple

class TdnnAffine(torch.nn.Module):
    """ An implemented tdnn affine component by conv1d
        y = splice(w * x, context) + b
    @input_dim: number of dims of frame <=> inputs channels of conv
    @output_dim: number of layer nodes <=> outputs channels of conv
    @context: a list of context
        e.g.  [-2,0,2]
    If context is [0], then the TdnnAffine is equal to linear layer.
    """
    def __init__(self, input_dim, output_dim, context=[0], bias=True, pad=True, stride=1, groups=1, norm_w=False, norm_f=False):
        super(TdnnAffine, self).__init__()
        assert input_dim % groups == 0
        # Check to make sure the context sorted and has no duplicated values
        for index in range(0, len(context) - 1):
            if(context[index] >= context[index + 1]):
                raise ValueError("Context tuple {} is invalid, such as the order.".format(context))

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context = context
        self.bool_bias = bias
        self.pad = pad
        self.groups = groups

        self.norm_w = norm_w
        self.norm_f = norm_f

        # It is used to subsample frames with this factor
        self.stride = stride

        self.left_context = context[0] if context[0] < 0 else 0 
        self.right_context = context[-1] if context[-1] > 0 else 0 

        self.tot_context = self.right_context - self.left_context + 1

        # Do not support sphereConv now.
        if self.tot_context > 1 and self.norm_f:
            self.norm_f = False
            print("Warning: do not support sphereConv now and set norm_f=False.")

        kernel_size = (self.tot_context,)

        self.weight = torch.nn.Parameter(torch.randn(output_dim, input_dim//groups, *kernel_size))

        if self.bool_bias:
            self.bias = torch.nn.Parameter(torch.randn(output_dim))
        else:
            self.register_parameter('bias', None)

        # init weight and bias. It is important
        self.init_weight()

        # Save GPU memory for no skiping case
        if len(context) != self.tot_context:
            # Used to skip some frames index according to context
            self.mask = torch.tensor([[[ 1 if index in context else 0 \
                                        for index in range(self.left_context, self.right_context + 1) ]]])
        else:
            self.mask = None

        ## Deprecated: the broadcast method could be used to save GPU memory, 
        # self.mask = torch.randn(output_dim, input_dim, 0)
        # for index in range(self.left_context, self.right_context + 1):
        #     if index in context:
        #         fixed_value = torch.ones(output_dim, input_dim, 1)
        #     else:
        #         fixed_value = torch.zeros(output_dim, input_dim, 1)

        #     self.mask=torch.cat((self.mask, fixed_value), dim = 2)

        # Save GPU memory of thi case.

        self.selected_device = False

    def init_weight(self):
        # Note, var should be small to avoid slow-shrinking
        torch.nn.init.normal_(self.weight, 0., 0.01)

        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.)


    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        # Do not use conv1d.padding for self.left_context + self.right_context != 0 case.
        if self.pad:
            inputs = F.pad(inputs, (-self.left_context, self.right_context), mode="constant", value=0)

        assert inputs.shape[2] >=  self.tot_context

        if not self.selected_device and self.mask is not None:
            # To save the CPU -> GPU moving time
            # Another simple case, for a temporary tensor, jus specify the device when creating it.
            # such as, this_tensor = torch.tensor([1.0], device=inputs.device)
            self.mask = to_device(self, self.mask)
            self.selected_device = True

        filters = self.weight  * self.mask if self.mask is not None else self.weight

        if self.norm_w:
            filters = F.normalize(filters, dim=1)

        if self.norm_f:
            inputs = F.normalize(inputs, dim=1)

        outputs = F.conv1d(inputs, filters, self.bias, stride=self.stride, padding=0, dilation=1, groups=self.groups)

        return outputs

    def extra_repr(self):
        return '{input_dim}, {output_dim}, context={context}, bias={bool_bias}, stride={stride}, ' \
               'pad={pad}, groups={groups}, norm_w={norm_w}, norm_f={norm_f}'.format(**self.__dict__)

    @classmethod
    def thop_count(self, m, x, y):
        x = x[0]

        kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
        bias_ops = 1 if m.bias is not None else 0

        # N x Cout x H x W x  (Cin x Kw x Kh + bias)
        total_ops = y.nelement() * (m.input_dim * kernel_ops + bias_ops)

        m.total_ops += torch.DoubleTensor([int(total_ops)])

class AttentionAlphaComponent(torch.nn.Module):
    """Compute the alpha with attention module.
            alpha = softmax(v'·f(w·x + b) + k) or softmax(v'·x + k)
    where f is relu here and bias could be lost.
    Support: 
            1. Single or Multi-head attention
            2. One affine or two affine
            3. Share weight (last affine = vector) or un-shared weight (last affine = matrix)
            4. Self-attention or time context attention (supported by context parameter of TdnnAffine)
            5. Different temperatures for different heads.
    """
    def __init__(self, input_dim, num_head=1, split_input=True, share=True, affine_layers=2, 
                 hidden_size=64, context=[0], bias=True, temperature=False, fixed=True):
        super(AttentionAlphaComponent, self).__init__()
        assert num_head >= 1
        # Multi-head case.
        if num_head > 1:
            if split_input:
                # Make sure fatures/planes with input_dim dims could be splited to num_head parts.
                assert input_dim % num_head == 0
            if temperature:
                if fixed:
                    t_list = []
                    for i in range(num_head):
                        t_list.append([[max(1, (i // 2) * 5)]])
                    # shape [1, num_head, 1, 1]
                    self.register_buffer('t', torch.tensor([t_list]))
                else:
                    # Different heads have different temperature.
                    # Use 1 + self.t**2 in forward to make sure temperature >= 1.
                    self.t = torch.nn.Parameter(torch.zeros(1, num_head, 1, 1))

        self.input_dim = input_dim
        self.num_head = num_head
        self.split_input = split_input
        self.share = share
        self.temperature = temperature
        self.fixed = fixed

        if share:
            # weight: [input_dim, 1] or [input_dim, hidden_size] -> [hidden_size, 1]
            final_dim = 1
        elif split_input:
            # weight: [input_dim, input_dim // num_head] or [input_dim, hidden_size] -> [hidden_size, input_dim // num_head]
            final_dim = input_dim // num_head
        else:
            # weight: [input_dim, input_dim] or [input_dim, hidden_size] -> [hidden_size, input_dim]
            final_dim = input_dim

        first_groups = 1
        last_groups = 1

        if affine_layers == 1:
            last_affine_input_dim = input_dim
            # (x, 1) for global case and (x, h) for split case.
            if num_head > 1 and split_input:
               last_groups = num_head
            self.relu_affine = False
        elif affine_layers == 2:
            last_affine_input_dim = hidden_size * num_head
            if num_head > 1:
                # (1, h) for global case and (h, h) for split case.
                last_groups = num_head
                if split_input:
                    first_groups = num_head
            # Add a relu-affine with affine_layers=2.
            self.relu_affine = True
            self.first_affine = TdnnAffine(input_dim, last_affine_input_dim, context=context, bias=bias, groups=first_groups)
            self.relu = torch.nn.ReLU(inplace=True)
        else:
            raise ValueError("Expected 1 or 2 affine layers, but got {}.",format(affine_layers))

        self.last_affine = TdnnAffine(last_affine_input_dim, final_dim * num_head, context=context, bias=bias, groups=last_groups)
        # Dim=2 means to apply softmax in different frames-index (batch is a 3-dim tensor in this case).
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        if self.temperature:
            batch_size = inputs.shape[0]
            chunk_size = inputs.shape[2]

        x = inputs
        if self.relu_affine:
            x = self.relu(self.first_affine(x))
        if self.num_head > 1 and self.temperature:
            if self.fixed:
                t = self.t
            else:
                t = 1 + self.t**2
            x = self.last_affine(x).reshape(batch_size, self.num_head, -1, chunk_size) / t
            return self.softmax(x.reshape(batch_size, -1, chunk_size))
        else:
            return self.softmax(self.last_affine(x))

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn

class DotProductAttention(nn.Module):
    """
    Compute the dot products of the query with all values and apply a softmax function to obtain the weights on the values
    """
    def __init__(self, hidden_dim):
        super(DotProductAttention, self).__init__()
        self.normalize = nn.LayerNorm(hidden_dim)
        self.out_projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, query: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(0)

        #  score = torch.bmm(query, value.transpose(1, 2))
        score = torch.matmul(query, value.transpose(1, 2))
        #  print(score.shape)
        #  attn = F.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        attn = F.softmax(score.view(-1, input_size), dim = 1)
        #  context = torch.bmm(attn, value)
        context = attn.matmul(value.squeeze(1))

        return context, attn


class AdditiveAttention(nn.Module):
    """
     Applies a additive attention (bahdanau) mechanism on the output features from the decoder.
     Additive attention proposed in "Neural Machine Translation by Jointly Learning to Align and Translate" paper.
     Args:
         hidden_dim (int): dimesion of hidden state vector
     Inputs: query, value
         - **query** (batch_size, q_len, hidden_dim): tensor containing the output features from the decoder.
         - **value** (batch_size, v_len, hidden_dim): tensor containing features of the encoded input sequence.
     Returns: context, attn
         - **context**: tensor containing the context vector from attention mechanism.
         - **attn**: tensor containing the alignment from the encoder outputs.
     Reference:
         - **Neural Machine Translation by Jointly Learning to Align and Translate**: https://arxiv.org/abs/1409.0473
    """
    def __init__(self, hidden_dim: int) -> None:
        super(AdditiveAttention, self).__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(hidden_dim, 1)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), value)
        return context, attn


class LocationAwareAttention(nn.Module):
    """
    Applies a location-aware attention mechanism on the output features from the decoder.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    The location-aware attention mechanism is performing well in speech recognition tasks.
    We refer to implementation of ClovaCall Attention style.
    Args:
        hidden_dim (int): dimesion of hidden state vector
        smoothing (bool): flag indication whether to use smoothing or not.
    Inputs: query, value, last_attn, smoothing
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **last_attn** (batch_size * num_heads, v_len): tensor containing previous timestep`s attention (alignment)
    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the feature from encoder outputs
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.
    Reference:
        - **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
        - **ClovaCall**: https://github.com/clovaai/ClovaCall/blob/master/las.pytorch/models/attention.py
    """
    def __init__(self, hidden_dim: int, smoothing: bool = True) -> None:
        super(LocationAwareAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score_proj = nn.Linear(hidden_dim, 1, bias=True)
        self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
        self.smoothing = smoothing

    def forward(self, query: Tensor, value: Tensor, last_attn: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, seq_len = query.size(0), query.size(2), value.size(1)

        # Initialize previous attention (alignment) to zeros
        if last_attn is None:
            last_attn = value.new_zeros(batch_size, seq_len)

        conv_attn = torch.transpose(self.conv1d(last_attn.unsqueeze(1)), 1, 2)
        score = self.score_proj(torch.tanh(
                self.query_proj(query.reshape(-1, hidden_dim)).view(batch_size, -1, hidden_dim)
                + self.value_proj(value.reshape(-1, hidden_dim)).view(batch_size, -1, hidden_dim)
                + conv_attn
                + self.bias
        )).squeeze(dim=-1)

        if self.smoothing:
            score = torch.sigmoid(score)
            attn = torch.div(score, score.sum(dim=-1).unsqueeze(dim=-1))
        else:
            attn = F.softmax(score, dim=-1)

        context = torch.bmm(attn.unsqueeze(dim=1), value).squeeze(dim=1)  # Bx1xT X BxTxD => Bx1xD => BxD

        return context, attn


class MultiHeadLocationAwareAttention(nn.Module):
    """
    Applies a multi-headed location-aware attention mechanism on the output features from the decoder.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    The location-aware attention mechanism is performing well in speech recognition tasks.
    In the above paper applied a signle head, but we applied multi head concept.
    Args:
        hidden_dim (int): The number of expected features in the output
        num_heads (int): The number of heads. (default: )
        conv_out_channel (int): The number of out channel in convolution
    Inputs: query, value, prev_attn
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **prev_attn** (batch_size * num_heads, v_len): tensor containing previous timestep`s attention (alignment)
    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the feature from encoder outputs
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.
    Reference:
        - **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
        - **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8, conv_out_channel: int = 10) -> None:
        super(MultiHeadLocationAwareAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim = int(hidden_dim / num_heads)
        self.conv1d = nn.Conv1d(num_heads, conv_out_channel, kernel_size=3, padding=1)
        self.loc_proj = nn.Linear(conv_out_channel, self.dim, bias=False)
        self.query_proj = nn.Linear(hidden_dim, self.dim * num_heads, bias=False)
        self.value_proj = nn.Linear(hidden_dim, self.dim * num_heads, bias=False)
        self.score_proj = nn.Linear(self.dim, 1, bias=True)
        self.out_projection = nn.Linear(hidden_dim << 1, hidden_dim, bias=True)
        self.bias = nn.Parameter(torch.rand(self.dim).uniform_(-0.1, 0.1))

    def forward(self, query: Tensor, value: Tensor, last_attn: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, seq_len = value.size(0), value.size(1)

        if last_attn is None:
            last_attn = value.new_zeros(batch_size, self.num_heads, seq_len)

        loc_energy = torch.tanh(self.loc_proj(self.conv1d(last_attn).transpose(1, 2)))
        loc_energy = loc_energy.unsqueeze(1).repeat(1, self.num_heads, 1, 1).view(-1, seq_len, self.dim)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.dim).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.dim).permute(0, 2, 1, 3)
        query = query.contiguous().view(-1, 1, self.dim)
        value = value.contiguous().view(-1, seq_len, self.dim)

        score = self.score_proj(torch.tanh(value + query + loc_energy + self.bias)).squeeze(2)
        attn = F.softmax(score, dim=1)

        value = value.view(batch_size, seq_len, self.num_heads, self.dim).permute(0, 2, 1, 3)
        value = value.contiguous().view(-1, seq_len, self.dim)

        context = torch.bmm(attn.unsqueeze(1), value).view(batch_size, -1, self.num_heads * self.dim)
        attn = attn.view(batch_size, self.num_heads, -1)

        return context, attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention proposed in "Attention Is All You Need"
    Instead of performing a single attention function with d_model-dimensional keys, values, and queries,
    project the queries, keys and values h times with different, learned linear projections to d_head dimensions.
    These are concatenated and once again projected, resulting in the final values.
    Multi-head attention allows the model to jointly attend to information from different representation
    subspaces at different positions.
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) 揃 W_o
        where head_i = Attention(Q 揃 W_q, K 揃 W_k, V 揃 W_v)
    Args:
        d_model (int): The dimension of keys / values / quries (default: 512)
        num_heads (int): The number of attention heads. (default: 8)
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): In transformer, three different ways:
            Case 1: come from previoys decoder layer
            Case 2: come from the input embedding
            Case 3: come from the output embedding (masked)
        - **key** (batch, k_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)
        - **value** (batch, v_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)
        - **mask** (-): tensor containing indices to be masked
    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features.
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)
        self.query_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.key_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.value_proj = nn.Linear(d_model, self.d_head * num_heads)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)  # BxQ_LENxNxD
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head)      # BxK_LENxNxD
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head)  # BxV_LENxNxD

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxQ_LENxD
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)      # BNxK_LENxD
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxV_LENxD

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # BxNxQ_LENxK_LEN

        context, attn = self.scaled_dot_attn(query, key, value, mask)

        context = context.view(self.num_heads, batch_size, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)  # BxTxND

        return context, attn


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout
    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked
    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """
    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 16,
            dropout_p: float = 0.1,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            pos_embedding: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._compute_relative_positional_encoding(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _compute_relative_positional_encoding(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score


class CustomizingAttention(nn.Module):
    r"""
    Customizing Attention
    Applies a multi-head + location-aware attention mechanism on the output features from the decoder.
    Multi-head attention proposed in "Attention Is All You Need" paper.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    I combined these two attention mechanisms as custom.
    Args:
        hidden_dim (int): The number of expected features in the output
        num_heads (int): The number of heads. (default: )
        conv_out_channel (int): The dimension of convolution
    Inputs: query, value, last_attn
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **last_attn** (batch_size * num_heads, v_len): tensor containing previous timestep`s alignment
    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch * num_heads, v_len): tensor containing the alignment from the encoder outputs.
    Reference:
        - **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
        - **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, conv_out_channel: int = 10) -> None:
        super(CustomizingAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim = int(hidden_dim / num_heads)
        self.scaled_dot_attn = ScaledDotProductAttention(self.dim)
        self.conv1d = nn.Conv1d(1, conv_out_channel, kernel_size=3, padding=1)
        self.query_proj = nn.Linear(hidden_dim, self.dim * num_heads, bias=True)
        self.value_proj = nn.Linear(hidden_dim, self.dim * num_heads, bias=False)
        self.loc_proj = nn.Linear(conv_out_channel, self.dim, bias=False)
        self.bias = nn.Parameter(torch.rand(self.dim * num_heads).uniform_(-0.1, 0.1))

    def forward(self, query: Tensor, value: Tensor, last_attn: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, q_len, v_len = value.size(0), query.size(1), value.size(1)

        if last_attn is None:
            last_attn = value.new_zeros(batch_size * self.num_heads, v_len)

        loc_energy = self.get_loc_energy(last_attn, batch_size, v_len)  # get location energy

        query = self.query_proj(query).view(batch_size, q_len, self.num_heads * self.dim)
        value = self.value_proj(value).view(batch_size, v_len, self.num_heads * self.dim) + loc_energy + self.bias

        query = query.view(batch_size, q_len, self.num_heads, self.dim).permute(2, 0, 1, 3)
        value = value.view(batch_size, v_len, self.num_heads, self.dim).permute(2, 0, 1, 3)
        query = query.contiguous().view(-1, q_len, self.dim)
        value = value.contiguous().view(-1, v_len, self.dim)

        context, attn = self.scaled_dot_attn(query, value)
        attn = attn.squeeze()

        context = context.view(self.num_heads, batch_size, q_len, self.dim).permute(1, 2, 0, 3)
        context = context.contiguous().view(batch_size, q_len, -1)

        return context, attn

    def get_loc_energy(self, last_attn: Tensor, batch_size: int, v_len: int) -> Tensor:
        conv_feat = self.conv1d(last_attn.unsqueeze(1))
        conv_feat = conv_feat.view(batch_size, self.num_heads, -1, v_len).permute(0, 1, 3, 2)

        loc_energy = self.loc_proj(conv_feat).view(batch_size, self.num_heads, v_len, self.dim)
        loc_energy = loc_energy.permute(0, 2, 1, 3).reshape(batch_size, v_len, self.num_heads * self.dim)

        return loc_energy

class MonoHeadAttention(nn.Module):
    def __init__(self, hidden_size, input_size = 1, hidden_layer = True, cuda = True, bias = False):
        super(MonoHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.activation = nn.ReLU()
        self.hidden_layer = hidden_layer
        self.bias = bias
        if hidden_layer:
            self.W = nn.Parameter(torch.Tensor(hidden_size, input_size), requires_grad = True)
            if bias:
                self.b = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad = True)
        self.v = nn.Parameter(torch.Tensor(hidden_size, 1), requires_grad = True)
        if bias:
            self.k = nn.Parameter(torch.Tensor(1, 1), requires_grad = True)
        self._initialize_parameters()
        #  if cuda:
        #      self._to_cuda()

    def _to_cuda(self):
        if self.hidden_layer:
            self.W = self.W.cuda()
            if self.bias:
                self.b = self.b.cuda()
        self.v = self.v.cuda()
        if self.bias:
            self.k = self.k.cuda()

    def _initialize_parameters(self):
        for parameter in self.parameters():
            nn.init.kaiming_normal_(parameter)
            
    def get_attention(self, x):
        '''
        params:
            x: (B, T, F)
        return:
            alpha: (B, 1, T)
        '''
        if self.hidden_layer:
            x = x.permute(0, 2, 1) # B, F, T
            if self.bias:
                hidden_mat = self.W.matmul(x).permute(0, 2, 1) + self.b # B, T, H
            else:
                hidden_mat = self.W.matmul(x).permute(0, 2, 1) # B, T, H
            x = self.activation(hidden_mat)
        if self.bias:
            e = x.matmul(self.v) + self.k # B, T, 1
        else:
            e = x.matmul(self.v)
        e = e.squeeze(-1) # B, T
        alpha = F.softmax(torch.tanh(e), dim = 1) # B, T
        return alpha.unsqueeze(-1) 
    
    def forward(self, x):
        '''
        params:
            x: (B, F, T)
        return:
            attention_mean: (B, F)
        '''
        x = x.permute(0, 2, 1) # B, T, F
        alpha = self.get_attention(x) # B, T, 1
        #  print(alpha)
        x = x.permute(0, 2, 1) # B, F, T
        attention_mean = x.matmul(alpha).squeeze(-1)
        return attention_mean

if __name__ == "__main__":
    mha = AttentionAlphaComponent(256, num_head = 8, hidden_size = 64)
    #  import torch
    inputs = torch.randn(5, 256, 4)
    outputs = mha(inputs)
    print(outputs.shape) # 5, 8, 4
    inputs = inputs.view(5, 8, -1, 4)
    print(inputs.shape)
    outputs = inputs.matmul(outputs.unsqueeze(-1))
    print(outputs.shape)
    #  attention = DotProductAttention(256)
    #  query = torch.randn(1, 1, 256)
    #  value = torch.randn(4, 1, 256)
    #  outputs, att = attention(query, value)
    outputs = outputs.view(5, 256, -1)
    print(outputs.shape)
    #  print(outputs.mean(dim = 1))
