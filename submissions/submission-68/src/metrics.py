import numpy as np
import sys

import torch
import torch.nn.functional as F


# ---------------------- generate and evaluate ----------------------------------------------------------------
@torch.no_grad()
def generate_and_evaluate(
    model,
    input_sequences,  # shape: (B, L)
    kb_assignments,  # list of length B of tuples
    knowledge_vocab_list,  # shape: (V,)
    V,
    pos_tmpls=None, # list of length B scalars that mark the postion of bi in each seq???
    max_gen_len=20,
    transition_dict=None,
    tmpl_list=None
):
    chain_list = [tmpl[0] for tmpl in tmpl_list] 
    
    B, L = input_sequences.shape
    device = input_sequences.device
    # completions = model.generate(input_sequences, max_new_tokens= max_gen_len)
    completions, gen_logits = model.generate(input_sequences, max_new_tokens=max_gen_len, return_logits=True) # gen_logits BxGxV
    completions_gen = completions[:, L:]  # (B, G)

    knowledge_mask = torch.zeros(V, dtype=torch.bool, device=device)
    knowledge_mask[knowledge_vocab_list] = True  # Mask out everything not in the list
    markov_mask = ~knowledge_mask  # Tokens not in knowledge
    
    b_indices = torch.tensor([k[1] for k in kb_assignments], device=device)
   
    # (2) At pos_list location:
    at_pos_is_kb_token = torch.zeros(B, dtype=torch.bool, device=device)
    at_pos_is_bi_token = torch.zeros(B, dtype=torch.bool, device=device)
    non_kb_outside_reserved = torch.zeros(B, device=device)
    
    at_pos_kb_loss = torch.zeros(B, device=device)
    at_pos_bi_loss = torch.zeros(B, device=device)
    non_kb_outside_loss = torch.zeros(B, device=device)

    if pos_tmpls is not None:
        probs = F.softmax(gen_logits, dim=-1)  # (B, G, V)
        for i in range(B):
            pos = pos_tmpls[i]
            tmpl_id = tmpl_list[i] 

          # ----------------------------------------------------------------------------------------------------------------

            token = completions[i, pos]
            at_pos_is_kb_token[i] = knowledge_mask[token]
            at_pos_is_bi_token[i] = (token == b_indices[i])

            mask = torch.ones(completions_gen.shape[1], dtype=torch.bool, device=device)
            reserved_pos_gen = pos_tmpls[i] - L
            if 0 <= reserved_pos_gen < completions_gen.shape[1]:
                mask[reserved_pos_gen] = False  # exclude the reserved position

            # All positions except reserved one
            tokens_to_check = completions_gen[i][mask]
            non_kb_flags = ~knowledge_mask[tokens_to_check]  # True if not a knowledge token
            non_kb_outside_reserved[i] = non_kb_flags.sum()
            #
            probs_i = probs[i]  # (G, V)
            probs_to_check = probs_i[mask]  # (G-1, V)

            # Sum over markov vocab
            p_markov_sum = probs_to_check[:, markov_mask].sum(dim=1)  # (G-1,)
            # Prevent log(0)
            ce_loss_per_pos = -torch.log(p_markov_sum + 1e-12)  # (G-1,)

            # Sum over positions
            non_kb_outside_loss[i] = ce_loss_per_pos.sum()


            ##########
            pos_gen = pos - L
            p = probs[i, pos_gen]  # (V,)
            # log prob of correct b_i
            bi_token = b_indices[i]
            at_pos_bi_loss[i] = -torch.log(p[bi_token] + 1e-12)
            # log prob of all knowledge tokens
            p_kb_sum = p[knowledge_vocab_list].sum()
            at_pos_kb_loss[i] = -torch.log(p_kb_sum + 1e-12)
    

    
    avg_kl_GT = kl_across_completion_GT(
                        prompt=input_sequences,
                        completions=completions,
                        gen_logits=gen_logits,
                        knowledge_vocab_list=knowledge_vocab_list,
                        vocab_size=V,
                        transition_dict = transition_dict,
                        chain_list = chain_list,  
                    )
    
    metrics =  {
                "at_pos_is_kb_rate": torch.mean(at_pos_is_kb_token.float()).item(),
                "at_pos_is_bi_rate": torch.mean(at_pos_is_bi_token.float()).item(),
                "non_kb_token_count_outside_reserved": torch.mean(non_kb_outside_reserved).item(),
                "at_pos_bi_loss": torch.mean(at_pos_bi_loss).item(),
                "at_pos_kb_loss": torch.mean(at_pos_kb_loss).item(),
                "non_kb_outside_reserved_loss": torch.mean(non_kb_outside_loss).item(),
                
                "kl_masked_completion_GT": avg_kl_GT['gen_kl_masked'],
                # "kl_unmasked_completion_GT": avg_kl_GT['gen_kl_unmasked'],
 
            }

    return metrics


def kl_across_completion_GT(
    prompt,         # (B, L) tensor
    completions,    # (B, L+G) tensor
    gen_logits,     # (B, G, V)
    knowledge_vocab_list,
    vocab_size,
    transition_dict = None,     # should be a dict of possible transition matrices if 'gt'
    chain_list = None,           # list of chain tmpls if 'gt'
):
    device = completions.device
    B, L_plus_G = completions.shape
    G = gen_logits.shape[1]
    L = int(L_plus_G - G)

    if chain_list[0] == -1:
        return {
                "gen_kl_masked": None,
                # "gen_kl_unmasked": None,
                # "gen_ce_masked": None,
                # "gen_ce_unmasked": None,
            }

    # Build markov vocab
    markov_vocab_list = [i for i in range(vocab_size) if i not in knowledge_vocab_list]
    V_markov = len(markov_vocab_list)

    token_to_index = {token_id: i for i, token_id in enumerate(markov_vocab_list)}
    knowledge_set = set(knowledge_vocab_list)
    markov_set = set(markov_vocab_list)

    # GT
    transition_probs = torch.stack([
            torch.tensor(transition_dict[tmpl], dtype=torch.float32, device=device)
            for tmpl in chain_list
        ])

    kl_masked_list = []
    kl_unmasked_list = []

    ce_masked_list, ce_unmasked_list = [], []

    for i in range(B):

        # Remove knowledge tokens from the full sequence and generation
        # Mask knowledge tokens from the full sequence
        is_kb_token = torch.tensor(
            [tok.item() in knowledge_set for tok in completions[i]], device=device
        )
        # Cleaned full sequence
        # print(completions[i])
        seq_full_clean = completions[i][~is_kb_token]
        # print(seq_full_clean)
        # print(L)
        num_kb_in_completion = is_kb_token[L:].sum().item()
        G_clean = G - num_kb_in_completion
        seq_gen_clean = seq_full_clean[-G_clean:]
        logits_mask = ~is_kb_token[L:]
        logits_seq = gen_logits[i][logits_mask]  # shape: (G_clean, V)   
        

        # seq_full = completions[i]
        # seq_gen = seq_full[-G:]
        # logits_seq = gen_logits[i]  # (G, V)
        L_clean = (~is_kb_token[:L]).sum().item()
        prev_token = seq_full_clean[L_clean - 1]
        # prev_token = seq_full_clean[L - 1]

        for t in range(G_clean):
            
            prev_idx = token_to_index[prev_token.item()]
            target_markov = transition_probs[i, prev_idx] + 1e-12
            target_markov /= target_markov.sum()               
            
            # Masked KL
            logits_masked = logits_seq[t].clone()
            logits_masked[knowledge_vocab_list] = float('-inf')
            model_probs_masked = F.softmax(logits_masked, dim=-1) + 1e-12
            model_probs_masked = model_probs_masked[markov_vocab_list]  # reorder to match target
            kl_m = F.kl_div(model_probs_masked.log(), target_markov, reduction='sum')
            kl_masked_list.append(kl_m)

            prev_token = seq_gen_clean[t]

    def avg_safe(tensor_list):
        return torch.mean(torch.stack(tensor_list)).item() if tensor_list else None

    return {
        "gen_kl_masked": avg_safe(kl_masked_list),
        # "gen_kl_unmasked": avg_safe(kl_unmasked_list),
        # "gen_ce_masked": avg_safe(ce_masked_list),
        # "gen_ce_unmasked": avg_safe(ce_unmasked_list),
    }


