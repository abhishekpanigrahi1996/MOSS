"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect

import tiktoken
import torch
import sys
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F

from collections import Counter



class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        pass

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        # return input



class CausalSelfAttention(nn.Module):

    def __init__(self, id, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.id = id
 
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.context_length, config.context_length))
                                        .view(1, 1, config.context_length, config.context_length))

        self.memory = config.memory
        self.device = config.device
    
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
       
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # Memory attention mask  -- 
            if self.memory >= 0:
                M1 = torch.ones(T, T, dtype=torch.bool).tril(diagonal=0)
                M2 = torch.ones(T, T, dtype=torch.bool).tril(diagonal=-self.memory-1)
                attn_mask = M1 * (~M2)
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask.to(self.device), dropout_p=self.dropout)
            else:
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        
     
        return y


class MLP(nn.Module):

    def __init__(self, id, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()
        self.id = id

        self.config = config
      
    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        
      
        return x


class Block(nn.Module):

    def __init__(self, id, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        if config.rpe: # relative PE
            self.attn = CausalSelfAttentionWithRPE(id, config)
        else: # absolute PE
            self.attn = CausalSelfAttention(id, config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(id, config)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x
    

class GPTBase(nn.Module):

    def __init__(self, config, pad_idx=None):
        super().__init__()
        assert config.vocab_size is not None
        assert config.context_length is not None
        self.config = config
        
        self.pad_idx = pad_idx

        if self.config.rpe:

            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=self.pad_idx),
                # wpe = nn.Embedding(config.context_length, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([Block(id, config) for id in range(config.n_layer)]),
                ln_f = LayerNorm(config.n_embd, bias=config.bias),
            ))

        else:
        
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=self.pad_idx),
                wpe = nn.Embedding(config.context_length, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([Block(id, config) for id in range(config.n_layer)]),
                ln_f = LayerNorm(config.n_embd, bias=config.bias),
            ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True) # changed! * 2
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        if self.config.vocab_size != 2:
            if not self.config.no_tying:
                print('tying embeddings!!!')
                self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        # print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())

        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, attn_mask=None, get_logits=False, loss_reduction='mean', pad_idx=-1, per_len=False):
        device = idx.device
        b, t = idx.size()
        
        assert t <= self.config.context_length, f"Cannot forward sequence of length {t}, block size is only {self.config.context_length}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        if not self.config.rpe:
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)

        if self.config.rpe:
            x = self.transformer.drop(tok_emb)
        else:
            x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x, attn_mask)
        x = self.transformer.ln_f(x) # (b, t, n_embd)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x) # (b, t, vocab_size)

            if not per_len:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), 
                                    ignore_index=pad_idx, reduction=loss_reduction)
   
                num_samples = (targets != pad_idx).sum().item()
                loss_per_len = None
                num_samples_per_len = None
            else:
                loss_per_token = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), 
                                    ignore_index=pad_idx, reduction='none')   # (batch_size * seq_len)
                loss_per_token = loss_per_token.view(targets.size(0), targets.size(1))  # (batch_size, seq_length)
                # Apply attention mask to zero out PAD tokens
                loss_per_token = loss_per_token * attn_mask  # (batch_size, seq_length)
                loss_per_len = loss_per_token.sum(dim=0)     # (seq_len,)
                num_samples_per_len = attn_mask.sum(dim=0)   # (seq_len,)
                loss = loss_per_len.sum()
                num_samples = num_samples_per_len.sum()

                if loss_reduction == 'mean':
                    loss = loss / num_samples

                # padding to be a tensor of size (context_window,)
                padding_length = self.config.context_length - loss_per_len.size(0)
                loss_per_len = F.pad(loss_per_len, (0, padding_length))
                num_samples_per_len = F.pad(num_samples_per_len, (0, padding_length))                



        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim # b x 1 x V
            # print(logits.shape)
            loss = None
            num_samples = None
            loss_per_len = None
            num_samples_per_len = None

        logits = logits if get_logits else None

        return {'logits': logits, 
                'loss': loss, 'num_samples': num_samples, 
                'loss_per_len': loss_per_len, 'num_samples_per_len': num_samples_per_len}
    
    

    def get_parameter_group_specs(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        param_decay = []
        param_no_decay = []    

        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                    param_no_decay.append(p)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    if self.config.vocab_size != 2:
                        if not self.config.no_tying:
                            if fpn == 'lm_head.weight':
                                continue
                    decay.add(fpn)
                    param_decay.append(p)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                    param_no_decay.append(p)
                
                elif pn.endswith('embk') or pn.endswith('embv'):
                    # new relative positional embeddings will NOT be weight decayed
                    no_decay.add(fpn)
                    param_no_decay.append(p)
                    

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}

        inter_params = decay & no_decay
        union_params = decay | no_decay

        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        param_no_decay = list({id(p): p for p in param_no_decay}.values())


        # —— helper to map tensor‑id → readable name ——————————
        name_lookup = {id(p): n for n, p in self.named_parameters()}

        decay_ids    = [id(p) for p in param_decay]
        no_decay_ids = [id(p) for p in param_no_decay]

        # 1) duplicates inside each list
        dupes_decay    = [pid for pid, c in Counter(decay_ids).items()    if c > 1]
        dupes_no_decay = [pid for pid, c in Counter(no_decay_ids).items() if c > 1]

        # 2) duplicates across the two groups
        cross_dupes = set(decay_ids) & set(no_decay_ids)

        # print(f"⇢ {len(dupes_decay)} dupes *within* decay group")
        # for pid in dupes_decay:
        #     print("  decay:", name_lookup[pid])

        # print(f"⇢ {len(dupes_no_decay)} dupes *within* no‑decay group")
        # for pid in dupes_no_decay:
        #     print("  no‑decay:", name_lookup[pid])

        # print(f"⇢ {len(cross_dupes)} dupes *across* groups")
        # for pid in cross_dupes:
        #     print("  both:", name_lookup[pid])

        return [
            {"params": param_decay},
            {"params": param_no_decay, "weight_decay": 0.0},
        ]


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, return_logits=False):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        logits_list = []

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at context_length
            idx_cond = idx if idx.size(1) <= self.config.context_length else idx[:, -self.config.context_length:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond, get_logits=True)['logits']
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

            if return_logits:
                logits_list.append(logits.unsqueeze(1))  # (B, 1, V)


        if return_logits:
            all_logits = torch.cat(logits_list, dim=1)  # (B, max_new_tokens, V)
            return idx, all_logits
        else:
            return idx
    
    @torch.no_grad()
    def generate_from_string(self, in_str, max_new_tokens, temperature=1.0, top_k=None):
        idx = torch.tensor(self.tokenizer.encode(in_str, allowed_special={"<|endoftext|>"})).view(1,-1).to(self.lm_head.weight.device)
        out_idx = self.generate(idx, max_new_tokens, temperature, top_k).view(-1).to('cpu').numpy()
        return self.tokenizer.decode(out_idx)




### for relative postional encoding
class CausalSelfAttentionWithRPE(nn.Module):

    def __init__(self, id, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.id = id
    
        self.flash = False
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.context_length, config.context_length))
                                        .view(1, 1, config.context_length, config.context_length))

        self.memory = config.memory
        self.device = config.device
        
        # relative positional embeddings
        self.embk = nn.Parameter(0.02 * torch.randn(config.context_length, config.n_embd // self.n_head))
        self.embv = nn.Parameter(0.02 * torch.randn(config.context_length, config.n_embd // self.n_head))


    def forward(self, x, attn_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # Memory attention mask
            # if self.memory >= 0:
            #     M1 = torch.ones(T, T, dtype=torch.bool).tril(diagonal=0)
            #     M2 = torch.ones(T, T, dtype=torch.bool).tril(diagonal=-self.memory-1)
            #     attn_mask = M1 * (~M2)
            #     y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask.to(self.device), dropout_p=self.dropout)
            # else:
            #     y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
            pass 
        else:
            # manual implementation of attention
            
            # relative positional embeddings
            q2 = q.permute(2, 0, 1, 3).contiguous().view(T, B * self.n_head, C // self.n_head)
            relative_pos = torch.arange(T)[:, None] - torch.arange(T)[None, :]
            relative_pos = relative_pos.tril().int().to(x.device)
            rk = self.embk[relative_pos]
            rv = self.embv[relative_pos]

            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att1 = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att2 = q2 @ rk.transpose(1, 2)
            att2 = att2.transpose(0, 1).contiguous().view(B, self.n_head, T, T)
            att = att1 + att2
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            # att shape: bs x head x T x T
            # print(attn_mask)
            if attn_mask is not None:
                padding_mask = attn_mask.unsqueeze(1).unsqueeze(2)
                # print(padding_mask.shape)
                # print('----------------------------------')
                att = att.masked_fill(padding_mask == 0, float('-inf'))

  
            # # Apply padding mask
  
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y1 = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y2 = att.permute(2, 0, 1, 3).contiguous().view(T, B * self.n_head, T)
            y2 = y2 @ rv
            y2 = y2.transpose(0, 1).contiguous().view(B, self.n_head, T, C // self.n_head)
            y = y1 + y2
            y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        
  
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        

        return y

