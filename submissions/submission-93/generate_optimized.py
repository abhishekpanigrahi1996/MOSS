import torch
import numpy as np
import torch.nn.functional as F
import math
import functools
print = functools.partial(print, flush=True)


def add_gumbel_noise(logits, temperature):
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64).clamp(min=1e-9)
    if temperature == 0: return logits
    else: gumbel_noise = (-torch.log(noise)) ** temperature; return logits.exp() / (gumbel_noise + 1e-9)

def get_num_transfer_tokens_schedule(mask_index, steps):
    if steps <= 0: print(f"Warning: steps is {steps}. Returning empty schedule."); return torch.zeros(mask_index.shape[0], 0, device=mask_index.device, dtype=torch.int64)
    mask_num = mask_index.sum(dim=1, keepdim=True); steps_tensor = torch.tensor(steps, device=mask_index.device, dtype=torch.long)
    safe_steps = torch.where(mask_num.squeeze(1) > 0, steps_tensor, torch.tensor(1, device=mask_index.device, dtype=torch.long))
    base = mask_num // safe_steps.unsqueeze(1); remainder = mask_num % safe_steps.unsqueeze(1)
    num_transfer_tokens_schedule = (torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base)
    for i in range(mask_num.size(0)):
        rem = remainder[i].item(); end_idx = min(rem, steps)
        if rem > 0: num_transfer_tokens_schedule[i, :end_idx] += 1
    total_scheduled = num_transfer_tokens_schedule.sum(dim=1); mismatch_indices = torch.where(total_scheduled != mask_num.squeeze(1))[0]
    for i in mismatch_indices:
        current_sum = total_scheduled[i].item(); target_sum = mask_num[i, 0].item(); diff = target_sum - current_sum
        if diff!=0: print(f"Correcting schedule mismatch item {i}: diff={diff}")
        step_idx = steps - 1
        while diff != 0 and step_idx >= 0:
            adjustment = 1 if diff > 0 else -1
            if num_transfer_tokens_schedule[i, step_idx].item() + adjustment >= 0: num_transfer_tokens_schedule[i, step_idx] += adjustment; diff -= adjustment
            step_idx -= 1
        if diff != 0: print(f"Error: Could not fully correct schedule mismatch item {i}. Final diff: {diff}")
    return num_transfer_tokens_schedule

def make_step_rewards(logits, token_masks):
    if logits.shape[:-1] != token_masks.shape: print(f"Error: Logits shape {logits.shape[:-1]} vs mask shape {token_masks.shape}"); return [[]] * logits.shape[0]
    probabilities = F.softmax(logits, dim=-1); probabilities = probabilities * token_masks.unsqueeze(-1)
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]; valid_rows = sample[token_masks[i]]
        if valid_rows.numel() > 0:
            num_labels = valid_rows.shape[-1]
            if num_labels != 2: print(f"Warning: Expected PRM dim 2, got {num_labels}. Assuming last logit."); positive_probs = valid_rows[:, -1]
            else: positive_probs = valid_rows.view(-1, 2)[:, 1]
            all_scores_res.append(positive_probs.cpu().tolist())
        else: all_scores_res.append([])
    return all_scores_res

def calculate_backmasking_probs(block_scores, backmasking_alpha=5.0, min_prob=0.01):
    if not block_scores: return []
    scores = np.array(block_scores); inverted_scores = 1.0 - scores; probs = np.exp(backmasking_alpha * inverted_scores)
    probs = np.maximum(probs, min_prob); max_p, min_p = probs.max(), probs.min()
    if max_p <= min_p: return np.full_like(probs, min_prob)
    probs = min_prob + (1 - min_prob) * (probs - min_p) / (max_p - min_p)
    return probs

def get_backmasking_tokens(block_region_masks, block_probs, backmasking_intensity=0.5, x_shape=None):
    if x_shape is None or not block_region_masks: return torch.zeros(x_shape, dtype=torch.bool)
    batch_size = block_region_masks[0].shape[0]; seq_len = block_region_masks[0].shape[1]
    if len(block_region_masks) != len(block_probs): print(f"Error: Mismatch block masks ({len(block_region_masks)}) vs probs ({len(block_probs)})."); return torch.zeros(x_shape, dtype=torch.bool)
    final_mask = torch.zeros(x_shape, dtype=torch.bool, device=block_region_masks[0].device)
    for i, (region_mask, prob) in enumerate(zip(block_region_masks, block_probs)):
        for b in range(batch_size):
            block_token_indices = torch.where(region_mask[b])[0]; block_size = len(block_token_indices);
            if block_size == 0: continue
            num_to_mask = int(block_size * prob * backmasking_intensity)
            if num_to_mask > 0:
                perm = torch.randperm(block_size, device=region_mask.device); mask_positions_in_block = block_token_indices[perm[:num_to_mask]]
                valid_indices = mask_positions_in_block[mask_positions_in_block < seq_len]; final_mask[b, valid_indices] = True
    return final_mask

def compute_block_score(block_text, prompt_text, prm_model, prm_tokenizer):
    # This function computes score for ONE block
    if not block_text.strip(): return 0.0
    formatted_block = "<extra_0>" + block_text + "<extra_0>"; system_prompt = "Please evaluate the reasoning step provided."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_text}, {"role": "assistant", "content": formatted_block}]
    try: conversation_str = prm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception as e: print(f"Error applying chat template: {e}"); return 0.0
    try:
        max_len = getattr(prm_tokenizer, 'model_max_length', 2048)
        batch = prm_tokenizer(conversation_str, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        input_ids = batch["input_ids"].to(prm_model.device); attention_mask = batch["attention_mask"].to(prm_model.device)
    except Exception as e: print(f"Error tokenizing for PRM: {e}"); return 0.0
    step_sep_id = prm_tokenizer.convert_tokens_to_ids("<extra_0>"); step_sep_id = step_sep_id[0] if isinstance(step_sep_id, list) else step_sep_id
    token_masks = torch.zeros_like(input_ids, dtype=torch.bool)
    indices = torch.where(input_ids[0] == step_sep_id)[0]
    if len(indices) > 0: token_masks[0, indices[-1]] = True
    else: print("Warning: <extra_0> token not found for scoring."); return 0.0
    try:
        with torch.no_grad(): outputs = prm_model(input_ids=input_ids, attention_mask=attention_mask)
        logits_or_scores = None
        if hasattr(outputs, 'logits'): logits_or_scores = outputs.logits
        elif hasattr(outputs, 'scores'): logits_or_scores = outputs.scores
        elif hasattr(outputs, 'end_scores'): logits_or_scores = outputs.end_scores
        elif isinstance(outputs, torch.Tensor): logits_or_scores = outputs
        else: print(f"Error: Unknown PRM output structure: {type(outputs)}"); return 0.0
        if logits_or_scores is None: print("Error: Failed to extract PRM output."); return 0.0
        # This assumes make_step_rewards can handle the extracted logits_or_scores structure
        step_reward = make_step_rewards(logits_or_scores[0:1], token_masks[0:1])
        if step_reward and step_reward[0]: avg_score = sum(step_reward[0]) / len(step_reward[0]); return max(0.0, min(1.0, avg_score))
        else: return 0.0
    except Exception as e: print(f"Error during PRM inference/score: {e}"); import traceback; traceback.print_exc(); return 0.0

def printable_sequence(x_tensor, tokenizer, mask_id, prompt_len, max_chars=None):
    """
    Render the decoded sequence for the first batch item, replacing every
    occurrence of `mask_id` with the literal string “[MASK]”.  The string is
    truncated to `max_chars` to keep the log readable.
    """
    toks = x_tensor[0, prompt_len:].tolist()
    pieces = []
    for t in toks:
        if t == mask_id:
            pieces.append("[MASK]")
        else:
            pieces.append(tokenizer.decode([t], skip_special_tokens=True))
    txt = "".join(pieces)
    if max_chars is not None and len(txt) > max_chars:
        return txt[:max_chars] + "..."
    return txt
def compute_k_block_scores_batched(
    x,
    start_block_index,       # index of first block in window
    num_blocks_in_window,    # number of blocks to score (K)
    prompt_len,
    block_length,
    prompt_text,             # text of the initial prompt
    tokenizer,
    prm_model,
    prm_tokenizer
):
    """
    Compute scores for K blocks in one batched forward pass.
    """
    # 1. Extract each block's text
    block_texts = []
    for i in range(num_blocks_in_window):
        idx = start_block_index + i
        start = prompt_len + idx * block_length
        end = start + block_length
        block_text = tokenizer.decode(x[0, start:end], skip_special_tokens=True)
        block_texts.append(block_text)

    # 2. Build one conversation string per block
    convs = []
    system_msg = {"role": "system", "content": "Please evaluate the reasoning step provided."}
    user_msg = {"role": "user",   "content": prompt_text}
    for block_text in block_texts:
        assistant_msg = {"role": "assistant", "content": f"<extra_0>{block_text}<extra_0>"}
        # Apply chat template for each individually (tokenize later as batch)
        conv_str = prm_tokenizer.apply_chat_template(
            [system_msg, user_msg, assistant_msg],
            add_generation_prompt=False,
            tokenize=False
        )
        convs.append(conv_str)

    # 3. Tokenize all conversations at once (padding/truncation ensures uniform shape)
    batch = prm_tokenizer(
        convs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=getattr(prm_tokenizer, "model_max_length", 2048)
    )
    input_ids = batch["input_ids"].to(prm_model.device)
    attention_mask = batch["attention_mask"].to(prm_model.device)

    # 4. Forward pass for all K blocks in a single call
    with torch.no_grad():
        outputs = prm_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = None
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif hasattr(outputs, "scores"):
            logits = outputs.scores
        elif hasattr(outputs, "end_scores"):
            logits = outputs.end_scores
        else:
            logits = outputs if isinstance(outputs, torch.Tensor) else None

    # 5. Build token masks: mark <extra_0> positions for each example
    sep_id = prm_tokenizer.convert_tokens_to_ids("<extra_0>")
    # If sep_id is list, take first
    sep_id = sep_id[0] if isinstance(sep_id, (list, tuple)) else sep_id
    token_masks = torch.zeros_like(input_ids, dtype=torch.bool)
    for b in range(input_ids.size(0)):
        idxs = (input_ids[b] == sep_id).nonzero(as_tuple=True)[0]
        if idxs.numel() > 0:
            token_masks[b, idxs[-1]] = True

    # 6. Compute step rewards in batch
    batch_rewards = make_step_rewards(logits, token_masks)  # returns list of lists

    # 7. Average rewards per block to get a score
    scores = []
    for rewards in batch_rewards:
        if rewards:
            avg = sum(rewards) / len(rewards)
            scores.append(float(max(0.0, min(1.0, avg))))
        else:
            scores.append(0.0)
    return scores


def demask_steps_refactored(x, mask_schedule, limit_mask, model, temperature, cfg_scale=0.0, remasking="low_confidence", mask_id=126336):
    steps = mask_schedule.shape[1]; prompt_len = 0; initial_non_masked = (x != mask_id);
    if initial_non_masked[0,0]:
        non_masked_indices = torch.where(initial_non_masked[0])[0]
        if len(non_masked_indices) > 0 and torch.all(non_masked_indices == torch.arange(len(non_masked_indices), device=x.device)): prompt_len = len(non_masked_indices)
    for i in range(steps):
        num_to_transfer_this_step = mask_schedule[:, i];
        if torch.all(num_to_transfer_this_step == 0): continue
        current_mask_index = (x == mask_id) & limit_mask
        if not current_mask_index.any(): break
        try:
            if cfg_scale > 0.0:
                cond_x = x.clone(); un_x = torch.full_like(x, mask_id)
                if prompt_len > 0: un_x[:, :prompt_len] = x[:, :prompt_len]
                x_in = torch.cat([cond_x, un_x], dim=0); logits_full = model(x_in).logits
                logits, un_logits = torch.chunk(logits_full, 2, dim=0); logits = logits + cfg_scale * (logits - un_logits)
            else: logits = model(x).logits
        except Exception as e: print(f"Error during model inference: {e}"); break
        if temperature == 0: x0 = torch.argmax(logits, dim=-1)
        else:
             logits_with_noise = add_gumbel_noise(logits, temperature)
             if torch.isinf(logits_with_noise).any() or torch.isnan(logits_with_noise).any(): print(f"Warning: NaN/Inf noisy logits step {i}. Using argmax."); x0 = torch.argmax(logits, dim=-1)
             else: x0 = torch.argmax(logits_with_noise, dim=-1)
        if remasking == "low_confidence": p = F.softmax(logits.to(torch.float64), dim=-1); x0_clamped = x0.clamp(0, p.shape[-1] - 1); x0_p = torch.gather(p, dim=-1, index=x0_clamped.unsqueeze(-1)).squeeze(-1)
        elif remasking == "random": x0_p = torch.rand_like(x0, dtype=torch.float64)
        else: raise NotImplementedError(remasking)
        confidence_for_selection = torch.where(current_mask_index, x0_p, torch.tensor(-np.inf, dtype=x0_p.dtype, device=x.device))
        batch_size = x.shape[0]
        for j in range(batch_size):
            num_to_transfer = num_to_transfer_this_step[j].item()
            if num_to_transfer > 0:
                available_masked = current_mask_index[j].sum().item(); k = min(num_to_transfer, available_masked)
                if k > 0:
                    try:
                        conf_j = confidence_for_selection[j]; select_indices = None
                        if conf_j.dim() == 0:
                           if conf_j > -np.inf: select_indices = torch.where(current_mask_index[j])[0]
                        elif k > conf_j.shape[0]: k = conf_j.shape[0]
                        if k > 0 and select_indices is None: _, select_indices = torch.topk(conf_j, k=k, largest=True)
                        if select_indices is not None and len(select_indices) > 0: x[j, select_indices] = x0[j, select_indices]
                    except RuntimeError as e: print(f"Error in topk item {j}: {e}"); continue
# --- End Helper Functions ---


# --- NEW HELPER: Compute Scores for a Window of K Blocks ---
def compute_k_block_scores(
    x,
    start_block_index, # The index of the first block in the window (0-based)
    num_blocks_in_window,
    prompt_len,
    block_length,
    prompt_text, # Assumes prompt_text is for batch item 0
    tokenizer,
    prm_model,
    prm_tokenizer
):
    """Computes scores for a specific window of K blocks."""
    scores = []
    print(f"Scoring blocks {start_block_index + 1} to {start_block_index + num_blocks_in_window}...")
    for i in range(num_blocks_in_window):
        block_idx = start_block_index + i
        b_start = prompt_len + block_idx * block_length
        b_end = b_start + block_length

        # Check bounds for the first batch item (scoring assumes batch size 1)
        if b_end > x.shape[1]:
             print(f"Warning: Block index {block_idx+1} exceeds sequence length during scoring.")
             scores.append(0.0) # Append default score
             continue

        # Decode only the first batch item for scoring
        block_text = tokenizer.decode(x[0, b_start:b_end], skip_special_tokens=True)
        block_score = compute_block_score(block_text, prompt_text, prm_model, prm_tokenizer)
        scores.append(block_score)
    return scores
# --- End Helper Functions ---

# --- NEW HELPER: Batched Scoring for Multiple Samples ---
def compute_k_block_scores_multi_sample(
    x_batch, # Shape: (num_samples * original_batch_size, seq_len)
    start_block_index,
    num_blocks_in_window,
    prompt_len,
    block_length,
    prompt_texts, # List of prompts, one per original batch item
    tokenizer,
    prm_model,
    prm_tokenizer,
    original_batch_size # Need this to group results
):
    """Computes scores for K blocks for multiple independent samples in a batch."""
    num_total_samples = x_batch.shape[0]
    if num_total_samples % original_batch_size != 0:
        print(f"Error: num_total_samples ({num_total_samples}) not divisible by original_batch_size ({original_batch_size})")
        # Return dummy scores matching expected output structure: List[List[List[float]]]
        # Outer list: original batch items, Middle list: samples per item, Inner list: K scores
        num_samples_per_item = num_total_samples // original_batch_size if original_batch_size > 0 else 1
        return [[[0.0] * num_blocks_in_window for _ in range(num_samples_per_item)] for _ in range(original_batch_size)]

    num_samples_per_item = num_total_samples // original_batch_size

    all_convs = []
    system_msg = {"role": "system", "content": "Please evaluate the reasoning step provided."}

    sample_indices = [] # Keep track of (original_batch_idx, sample_within_batch_idx)

    print(f"Scoring {num_total_samples} samples ({original_batch_size} original items x {num_samples_per_item} samples each) for {num_blocks_in_window} blocks...")

    for b_orig in range(original_batch_size):
        current_prompt_text = prompt_texts[b_orig]
        user_msg = {"role": "user", "content": current_prompt_text}
        for s in range(num_samples_per_item):
            sample_idx_global = b_orig * num_samples_per_item + s
            sample_indices.append((b_orig, s))

            for i in range(num_blocks_in_window):
                block_idx = start_block_index + i
                b_start = prompt_len + block_idx * block_length
                b_end = b_start + block_length

                # Decode block from the specific sample in the large batch
                block_text = tokenizer.decode(x_batch[sample_idx_global, b_start:b_end], skip_special_tokens=True)
                assistant_msg = {"role": "assistant", "content": f"<extra_0>{block_text}<extra_0>"}

                # Apply chat template for each block individually
                try:
                    conv_str = prm_tokenizer.apply_chat_template(
                        [system_msg, user_msg, assistant_msg],
                        add_generation_prompt=False,
                        tokenize=False
                    )
                    all_convs.append(conv_str)
                except Exception as e:
                    print(f"Error applying chat template for sample {sample_idx_global}, block {i}: {e}")
                    all_convs.append("") # Append empty string on error to maintain structure size

    if not all_convs:
        print("Error: No conversations generated for scoring.")
        # Return dummy scores matching expected output structure
        return [[[0.0] * num_blocks_in_window for _ in range(num_samples_per_item)] for _ in range(original_batch_size)]


    # Tokenize all conversations (num_total_samples * num_blocks_in_window) at once
    try:
        batch = prm_tokenizer(
            all_convs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=getattr(prm_tokenizer, "model_max_length", 2048)
        )
        input_ids = batch["input_ids"].to(prm_model.device)
        attention_mask = batch["attention_mask"].to(prm_model.device)
    except Exception as e:
        print(f"Error tokenizing for PRM batch scoring: {e}")
        # Return dummy scores
        return [[[0.0] * num_blocks_in_window for _ in range(num_samples_per_item)] for _ in range(original_batch_size)]


    # Single large forward pass for all blocks across all samples
    all_logits = None
    try:
        with torch.no_grad():
            outputs = prm_model(input_ids=input_ids, attention_mask=attention_mask)
            # Extract logits (handle different output formats)
            if hasattr(outputs, "logits"): all_logits = outputs.logits
            elif hasattr(outputs, "scores"): all_logits = outputs.scores
            elif hasattr(outputs, "end_scores"): all_logits = outputs.end_scores
            elif isinstance(outputs, torch.Tensor): all_logits = outputs
            else: raise ValueError("Unknown PRM output structure")
    except Exception as e:
        print(f"Error during batched PRM inference: {e}")
        # Return dummy scores
        return [[[0.0] * num_blocks_in_window for _ in range(num_samples_per_item)] for _ in range(original_batch_size)]

    if all_logits is None:
         print("Error: Failed to extract logits from PRM.")
         return [[[0.0] * num_blocks_in_window for _ in range(num_samples_per_item)] for _ in range(original_batch_size)]


    # Build token masks: mark <extra_0> positions for each example in the large batch
    sep_id = prm_tokenizer.convert_tokens_to_ids("<extra_0>")
    sep_id = sep_id[0] if isinstance(sep_id, (list, tuple)) else sep_id
    token_masks = torch.zeros_like(input_ids, dtype=torch.bool)
    # Find the *last* sep_id for each item in the large batch
    for b_idx in range(input_ids.size(0)):
        idxs = (input_ids[b_idx] == sep_id).nonzero(as_tuple=True)[0]
        if idxs.numel() > 0:
            token_masks[b_idx, idxs[-1]] = True
        # else: print warning? Might happen if block text was empty or template failed.

    # Compute step rewards for the entire large batch
    # make_step_rewards returns list of lists, length = input batch size (num_total_samples * K)
    all_block_rewards = make_step_rewards(all_logits, token_masks)

    # Reshape/Organize scores back into [original_batch][sample][block_score]
    # Expected structure: List[List[List[float]]]
    final_scores = [[[] for _ in range(num_samples_per_item)] for _ in range(original_batch_size)]
    current_reward_idx = 0
    for b_orig in range(original_batch_size):
        for s in range(num_samples_per_item):
            sample_scores = []
            for i in range(num_blocks_in_window):
                if current_reward_idx < len(all_block_rewards):
                    rewards = all_block_rewards[current_reward_idx]
                    if rewards:
                        avg = sum(rewards) / len(rewards)
                        sample_scores.append(float(max(0.0, min(1.0, avg))))
                    else:
                        sample_scores.append(0.0) # Default score if no reward computed
                    current_reward_idx += 1
                else:
                    print(f"Warning: Ran out of rewards at index {current_reward_idx}")
                    sample_scores.append(0.0) # Append default if out of bounds
            final_scores[b_orig][s] = sample_scores

    if current_reward_idx != len(all_block_rewards):
         print(f"Warning: Score processing consumed {current_reward_idx} rewards, but {len(all_block_rewards)} were generated.")

    return final_scores


@torch.no_grad()
def generate_prm_window_score( 
    model,
    prompt,
    prm_model,
    tokenizer,
    prm_tokenizer,
    steps=128,
    gen_length=512,
    block_length=32,
    temperature=0.3,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    backmasking_alpha=5.0,
    backmasking_intensity=0.5,
    backmasking_frequency=8, # K value
    backmasking_threshold=0.8,
    num_refinement_samples=12, # N value (replaces max_retry_attempts)
    selection_metric="product" # "product" or "min_score"
):
    print("===== Generation Started (PRM Window Score + Batch Refine) =====")
    if not isinstance(prompt, torch.Tensor) or prompt.dim() != 2: print("Error: Prompt must be 2D Tensor."); return None
    original_batch_size = prompt.shape[0]
    if gen_length <= 0 or block_length <= 0 or steps <= 0 or backmasking_frequency <= 0 or num_refinement_samples <= 0: print("Error: lengths, steps, frequency, samples > 0."); return None
    if gen_length % block_length != 0: print(f"Warning: gen_length not divisible by block_length.")
    num_blocks = gen_length // block_length; effective_gen_length = num_blocks * block_length
    if steps < num_blocks: print(f"Warning: steps ({steps}) < num_blocks ({num_blocks}).")
    block_steps_base = steps // num_blocks; remainder_steps = steps % num_blocks
    print(f"Batch Size: {original_batch_size}")
    print(f"Prompt (Item 0): {tokenizer.decode(prompt[0], skip_special_tokens=True)[:100]}...")
    print(f"Config: Steps={steps}, GenLen={effective_gen_length}, BlockLen={block_length}, K={backmasking_frequency}, Thresh={backmasking_threshold}, N_Samples={num_refinement_samples}, SelectMetric={selection_metric}")

    device = model.device; prompt_len = prompt.shape[1]
    x = torch.full((original_batch_size, prompt_len + effective_gen_length), mask_id, dtype=torch.long, device=device)
    x[:, : prompt_len] = prompt.clone()
    K = backmasking_frequency
    N = num_refinement_samples

    # Decode prompts for all items in the original batch
    prompt_texts = [tokenizer.decode(p, skip_special_tokens=True) for p in prompt]

    # Store scores per original batch item
    block_scores_all = [[0.0] * num_blocks for _ in range(original_batch_size)]
    # Store region masks (only need one set, applied to all samples)
    block_masks_list = []

    # --- Main Generation Loop ---
    for num_block in range(num_blocks):
        print(f"\n--- Generating Block {num_block+1}/{num_blocks} ---")
        block_start_idx = prompt_len + num_block * block_length
        block_end_idx = block_start_idx + block_length

        # --- 1. Generate Block Content (for the current state 'x') ---
        current_block_mask_bool = (x[:, block_start_idx:block_end_idx] == mask_id)
        if current_block_mask_bool.any():
            steps_this_block = block_steps_base + 1 if num_block < remainder_steps else block_steps_base
            if steps_this_block > 0:
                # Create mask schedule only for the portion to be generated
                schedule_mask_shape = torch.zeros_like(x, dtype=torch.bool)
                schedule_mask_shape[:, block_start_idx:block_end_idx] = current_block_mask_bool
                mask_schedule_block = get_num_transfer_tokens_schedule(schedule_mask_shape, steps_this_block)

                # Limit demasking to the current block for initial generation
                limit_mask_block = torch.zeros_like(x, dtype=torch.bool)
                limit_mask_block[:, :block_end_idx] = True # Allow attending to previous blocks

                # Apply demasking to the current state 'x'
                demask_steps_refactored(x, mask_schedule_block, limit_mask_block, model, temperature, cfg_scale, remasking, mask_id)
            else: print(f"Warning: 0 steps for block {num_block+1}. Skipping generation.")
        else: print(f"Block {num_block+1} already filled. Skipping generation.")

        # Store region mask for this block (shape: B, L)
        block_region_mask = torch.zeros_like(x, dtype=torch.bool)
        block_region_mask[:, block_start_idx:block_end_idx] = True
        block_masks_list.append(block_region_mask)

        # --- 2. Score Window & Potential Batch Refinement (Every K blocks) ---
        is_check_time = (num_block + 1) % K == 0 and K > 0
        if is_check_time:
            start_block_idx_window = num_block - K + 1
            print(f"\n--- Evaluating Window: Blocks {start_block_idx_window + 1} to {num_block + 1} ---")

            # --- Score the K blocks for the CURRENT state 'x' (original batch size) ---
            # Use the original batched scoring function here for the initial check
            current_window_scores_per_item = []
            # TODO: Optimize this initial scoring - can we reuse compute_k_block_scores_multi_sample?
            # For now, score each item separately or adapt compute_k_block_scores_batched for B>1
            print("Scoring current state (before potential refinement)...")
            temp_scores_container = compute_k_block_scores_multi_sample(
                 x, start_block_idx_window, K, prompt_len, block_length,
                 prompt_texts, tokenizer, prm_model, prm_tokenizer, original_batch_size
            )
            # temp_scores_container is List[List[List[float]]] with inner list size 1
            current_window_scores_per_item = [item_scores[0] for item_scores in temp_scores_container] # Extract the single sample scores


            # Update the main scores list for all batch items based on current state
            for b in range(original_batch_size):
                if current_window_scores_per_item and b < len(current_window_scores_per_item):
                     scores_b = current_window_scores_per_item[b]
                     if len(scores_b) == K:
                          for i in range(K):
                              block_scores_all[b][start_block_idx_window + i] = scores_b[i]
                     else: print(f"Warning: Incorrect score count for item {b} pre-refine.")


            # --- Check Threshold and Trigger Batch Refinement ---
            needs_refinement_flags = [False] * original_batch_size
            for b in range(original_batch_size):
                 if current_window_scores_per_item and b < len(current_window_scores_per_item):
                    min_score_b = min(current_window_scores_per_item[b]) if current_window_scores_per_item[b] else 0.0
                    if min_score_b < backmasking_threshold:
                        needs_refinement_flags[b] = True
                        print(f"Item {b}: Min score {min_score_b:.4f} < threshold ({backmasking_threshold:.4f}). Needs refinement.")
                    else:
                        print(f"Item {b}: Min score {min_score_b:.4f} >= threshold. OK.")
                 else: print(f"Warning: Missing scores for item {b}, cannot check threshold.")


            if any(needs_refinement_flags):
                print(f"--- Starting Batch Refinement for {sum(needs_refinement_flags)} items ---")

                # 1. Calculate backmasking probs based on *initial* scores (per item)
                backmasking_probs_per_item = []
                for b in range(original_batch_size):
                     scores_b = current_window_scores_per_item[b] if current_window_scores_per_item else []
                     # Use scores if needed refinement, otherwise dummy probs (won't be used)
                     probs_b = calculate_backmasking_probs(scores_b, backmasking_alpha) if needs_refinement_flags[b] else [0.0] * K
                     backmasking_probs_per_item.append(probs_b)

                # 2. Get backmasking tokens for the window (for the whole original batch)
                window_block_masks = block_masks_list[start_block_idx_window : num_block + 1] # List of K tensors (B, L)

                # Need to generate the mask based on per-item probabilities
                # get_backmasking_tokens expects List[Tensor(B,L)], List[List[float]]? No, probs is flat list.
                # Let's adapt: create one mask for the whole batch B, applying intensity only where needed.
                full_backmasking_mask = torch.zeros_like(x, dtype=torch.bool) # Shape (B, L)
                for b_idx in range(original_batch_size):
                    if needs_refinement_flags[b_idx]:
                         # Generate mask only for this item b_idx using its probs
                         item_mask = get_backmasking_tokens(
                              [bm[b_idx:b_idx+1] for bm in window_block_masks], # Slice masks for this item
                              backmasking_probs_per_item[b_idx],
                              backmasking_intensity,
                              x_shape=(1, x.shape[1]) # Shape for single item
                         )
                         # Apply this item's mask to the full batch mask
                         if item_mask is not None and item_mask.shape[0] == 1:
                              full_backmasking_mask[b_idx] = item_mask[0]

                num_backmasked = full_backmasking_mask.sum().item()
                if num_backmasked == 0:
                    print("No tokens selected for backmasking across batch. Skipping refinement.")
                    continue # Skip to next block generation phase

                print(f"Total tokens selected for backmasking across batch: {num_backmasked}")

                # 3. Create Batch for Refinement (N samples per original item)
                # repeat_interleave duplicates items sequentially: [item0_s0, item0_s1, ..., item1_s0, ...]
                x_batch = x.repeat_interleave(N, dim=0) # Shape (B*N, L)
                mask_batch = full_backmasking_mask.repeat_interleave(N, dim=0) # Shape (B*N, L)

                # 4. Apply Mask to Batch
                x_batch[mask_batch] = mask_id

                # 5. Batched Refinement (Demasking)
                print(f"Running batched refinement for {x_batch.shape[0]} samples...")
                # Limit refinement to the full sequence length (prompt+gen)
                refinement_mask_idx = (x_batch == mask_id) & (torch.arange(x_batch.shape[1], device=device).unsqueeze(0) >= prompt_len)
                # Use total number of steps = total masked tokens for simplicity (might be slow)
                # Alternative: fixed number of steps? Let's stick to full for now.
                # Calculate schedule for the large batch
                total_masked_in_batch = refinement_mask_idx.sum()
                # Ensure steps > 0 if there are masked tokens
                refinement_steps = max(1, total_masked_in_batch.item() // x_batch.shape[0]) if total_masked_in_batch > 0 else 0 # Avg steps per sample
                refinement_steps = min(refinement_steps, 128)
                print(f"Refinement using approx {refinement_steps} steps per sample.")

                if refinement_steps > 0:
                    mask_schedule_refine = get_num_transfer_tokens_schedule(refinement_mask_idx, refinement_steps)
                    window_token_start = prompt_len + start_block_idx_window * block_length
                    # The end index should be exclusive, covering up to the end of the *last* block in the window
                    window_token_end = prompt_len + (num_block + 1) * block_length # num_block is the *last* block index

                    # Limit refinement changes strictly to the current window
                    limit_mask_refine = torch.zeros_like(x_batch, dtype=torch.bool)
                    limit_mask_refine[:, prompt_len:window_token_end] = True # Corrected limit

                    print(f"Refinement limit mask set for tokens {window_token_start} to {window_token_end}") # Add logging

                    demask_steps_refactored(
                            x_batch,
                            mask_schedule_refine,
                            limit_mask_refine,
                        model,
                        temperature,
                        cfg_scale,
                        remasking,
                        mask_id,
                    )

                    print(f"Refined {int(refinement_mask_idx.sum().item())} tokens over "
                            f"{refinement_steps} steps for {x_batch.size(0)} samples.")
                else:
                    print("No refinement steps needed (or no tokens masked).")


                # 6. Batched Scoring of Refined Samples
                print("Scoring refined samples...")
                # all_refined_scores shape: List[List[List[float]]] -> [original_batch][sample][K scores]
                all_refined_scores = compute_k_block_scores_multi_sample(
                    x_batch, start_block_idx_window, K, prompt_len, block_length,
                    prompt_texts, tokenizer, prm_model, prm_tokenizer, original_batch_size
                )

                # 7. Select Best Sample for each original batch item
                print("Selecting best sample for each original batch item...")
                new_x = torch.zeros_like(x) # Create new tensor for the chosen states
                best_scores_chosen = [[0.0]*K for _ in range(original_batch_size)] # Store scores of chosen samples

                for b in range(original_batch_size):
                    if not needs_refinement_flags[b]:
                        # If this item didn't need refinement, keep original state and scores
                        new_x[b] = x[b]
                        best_scores_chosen[b] = block_scores_all[b][start_block_idx_window : start_block_idx_window + K]
                        continue

                    # This item needed refinement, find the best among its N samples
                    best_sample_idx_local = -1
                    best_metric_val = -float('inf')
                    valid_samples_found = False

                    if b < len(all_refined_scores):
                        samples_for_item_b = all_refined_scores[b] # List of N lists (K scores each)
                        for s_idx, sample_scores in enumerate(samples_for_item_b):
                            if not sample_scores: continue # Skip if scoring failed for this sample

                            current_metric_val = 0.0
                            if selection_metric == "product":
                                # Use small epsilon to avoid issues with zero scores
                                current_metric_val = math.prod(max(s, 1e-9) for s in sample_scores)
                            elif selection_metric == "min_score":
                                current_metric_val = min(sample_scores)
                            else: # Default to product
                                current_metric_val = math.prod(max(s, 1e-9) for s in sample_scores)

                            if current_metric_val > best_metric_val:
                                best_metric_val = current_metric_val
                                best_sample_idx_local = s_idx
                                valid_samples_found = True
                    else: print(f"Warning: Missing scores for original batch item {b} during selection.")


                    if valid_samples_found:
                        best_sample_idx_global = b * N + best_sample_idx_local
                        new_x[b] = x_batch[best_sample_idx_global]
                        best_scores_chosen[b] = all_refined_scores[b][best_sample_idx_local]
                        print(f"Item {b}: Chose sample {best_sample_idx_local} (Metric: {best_metric_val:.6f}, Scores: {[f'{s:.3f}' for s in best_scores_chosen[b]]})")
                    else:
                        # Fallback: Keep original if no valid refined sample found
                        print(f"Item {b}: No valid refined samples found. Keeping original state.")
                        new_x[b] = x[b]
                        best_scores_chosen[b] = block_scores_all[b][start_block_idx_window : start_block_idx_window + K]


                # 8. Update Overall State and Scores
                x = new_x # Update the main state tensor 'x'
                # Update the main scores list with the chosen scores for the window
                for b in range(original_batch_size):
                    for i in range(K):
                         block_scores_all[b][start_block_idx_window + i] = best_scores_chosen[b][i]

            # Else (if not any(needs_refinement_flags)): No refinement needed, scores already updated


    # --- Handle Final Partial Window (Scoring Only) ---
    remaining_blocks = num_blocks % K
    if remaining_blocks > 0 and num_blocks > 0:
        start_block_idx_final = num_blocks - remaining_blocks
        print(f"\n--- Scoring Final Partial Window: Blocks {start_block_idx_final + 1} to {num_blocks} ---")
        # Use the multi-sample scorer (with N=1 effectively, as we score the final 'x')
        final_scores_container = compute_k_block_scores_multi_sample(
             x, start_block_idx_final, remaining_blocks, prompt_len, block_length,
             prompt_texts, tokenizer, prm_model, prm_tokenizer, original_batch_size
        )
        final_window_scores_per_item = [item_scores[0] for item_scores in final_scores_container]


        # Update the main scores list for the final blocks
        for b in range(original_batch_size):
            if final_window_scores_per_item and b < len(final_window_scores_per_item):
                 scores_b = final_window_scores_per_item[b]
                 if len(scores_b) == remaining_blocks:
                     for i in range(remaining_blocks):
                         block_scores_all[b][start_block_idx_final + i] = scores_b[i]
                     print(f"Item {b} Final Window Scores: {[f'{s:.4f}' for s in scores_b]}")
                 else: print(f"Warning: Incorrect score count for item {b} in final window.")
            else: print(f"Warning: Missing final scores for item {b}.")


    # --- Final Output ---
    print("\n===== Generation Complete =====")
    for b in range(original_batch_size):
         final_masked_count = (x[b, prompt_len:] == mask_id).sum().item()
         if final_masked_count > 0: print(f"Warning (Item {b}): {final_masked_count} mask tokens remain.")
         print(f"Final block scores (Item {b}): {[f'{s:.3f}' for s in block_scores_all[b]]}")
         decoded = tokenizer.decode(x[b, prompt_len:], skip_special_tokens=True)
         print(f"\nGenerated output (Item {b}, first 500 chars):\n{decoded[:500]}\n{'...' if len(decoded) > 500 else ''}")

    return x

