import random
import numpy as np
import logging
import sys

from collections import Counter

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .utils import Logger, raise_and_log_error

class MarkovKnowledgeGenerator:
    def __init__(self, V: int, M: int, order: int, seq_length: int, special_toks: dict, logger: Logger, 
                 insertion_mode="shift", device: str = "cpu", skip_sepctok: bool = False, 
                 num_chain_tmpl: int=-1, initial: str='stationary', 
                 num_pos_tmpl: int = -1, bi_pos: str = "random", 
                 ood_frac: float = -1, chain_per_pos: int = -1
                 ):
        """
        A class to manage dataset generation with a knowledge base and Markov chain.

        Args:
            V (int): Total vocabulary size.
            M (int): Number of knowledge pairs
            Thus, 2M tokens for knowledge, V-2M for MC
            --------------------------------------------
            order (int): order of markov chain -- only supports order 0 and 1 for now
            seq_length (int): length of each input sequence (generates seqs of seq_length+1 to have seq_length tokens for input and seq_length tokens as next tokens)
            skip_sepctok: don't add bos and eos!
            device (str): 'cpu' or 'cuda'.
            --------------------------------------------
            num_chain_tmpl (int): if -1, for each sequence generate a fresh markov transition probability
                                  if >=1, choose from the finite "markov_set" number of transition matrices. 
                                  if =0, use random noise (unigram/bigram with uniform distribution)
            num_pos_tmpl: if -1 place knowledge pairs in random positions
                          if >=1 use a fixed number of position templates
            bi_pos: if "fixed", bi appears at the final position. if "random" is chosen randomly as ai.
            --------------------------------------------
            chain_per_pos: for the case of num_chain_tmpl and num_pos_tmpl are both >=0, measure of diversity within templates
            ood_frac: measure of diversity between factoids and templates
            --------------------------------------------
        """

        # ------------- general params ---------------------------------------------------------
        self.device = device
        self.skip_spectok = skip_sepctok
        self.special_toks = special_toks
        self.logger = logger

        self.seq_length = seq_length + 1
        
        self.V = V

        # ------------- knowledge params ---------------------------------------------------------
        self.M = M

        self.insertion_mode = insertion_mode

        # Generate knowledge base
        self.knowledge_pairs, self.knowledge_tokens = self.generate_knowledge_base()
        self.kp_dict = {pair: idx for idx, pair in enumerate(self.knowledge_pairs) }

        # The leftover vocabulary (for the Markov chain)
        all_tokens = set(range(V))
        self.chain_tokens = sorted(list(all_tokens - set(self.knowledge_tokens)))        
        self.vocab_size_for_chain = len(self.chain_tokens)                         

        # ------------- MC params ---------------------------------------------------------
        self.order = order
        if self.order > 2:
            raise_and_log_error(f'order > 2 MC not implemented yet.', err_type='value', logger=self.logger)
        
        self.initial = initial
        if self.initial not in ['uniform', 'stationary']:
            raise_and_log_error(err_str=f'MC starting from {self.initial} not implemented', err_type='value', logger=self.logger)

        # -------------- Templates ---------------------------------------------------------------
        # postition templates
        self.use_pos_tmpl = num_pos_tmpl != -1
        self.num_pos_tmpl = num_pos_tmpl if num_pos_tmpl > -1 else 0
        self.bi_pos = bi_pos

        self.pos_tmpls = self.generate_position_templates() if self.use_pos_tmpl else None

        # markov chain templates
        self.use_mc_tmpl = num_chain_tmpl > -1
        self.num_chain_tmpl = max(num_chain_tmpl, 1) if num_chain_tmpl > -1 else 0
        if (num_chain_tmpl == 0) and (self.order > 0):
            raise_and_log_error('only unigram supported with num_chain_tmpl=0', err_type='value', logger=self.logger)
        
        # ----------------------------------------------------------------------
        if self.use_mc_tmpl: # a dictionary of transition matrices
            self.transition_mat_dict = self.create_markov_transition_matrix(num_chain_tmpl=self.num_chain_tmpl, uniform=(num_chain_tmpl==0)) 
        else: 
            self.transition_mat_dict = None
            
        if (self.use_mc_tmpl) and (self.use_pos_tmpl):
            self.tmpl_type = 'mc-pos'
        elif (not self.use_mc_tmpl) and (self.use_pos_tmpl):
            self.tmpl_type = 'pos'
        elif (self.use_mc_tmpl) and (not self.use_pos_tmpl):
            self.tmpl_type = 'mc'
        elif (not self.use_mc_tmpl) and (not self.use_pos_tmpl):
            self.tmpl_type = None
        
        # -------------- in-dist / out-dist assignment ---------------------------------------------------------------
        self.use_ood = ood_frac != -1
        if self.use_ood and (self.tmpl_type is None):
            raise_and_log_error('choose another ood_frac', err_type="value", logger=self.logger)

        self.ood_frac = ood_frac
        self.chain_per_pos = chain_per_pos
        self.generate_in_dist_mask()

        # tracking which kp's appear with each mc and/or position (in-distribution)
        self.mem_dict = {}

        if self.tmpl_type == 'mc':
            for c in range(self.num_chain_tmpl):
                idxs = np.flatnonzero(self.in_dist_mask[c, :])
                self.mem_dict[f"mc_{c}"] = sorted([self.knowledge_pairs[i][1] for i in idxs])

        elif self.tmpl_type == 'pos':
            for p in range(self.num_pos_tmpl):
                idxs = np.flatnonzero(self.in_dist_mask[p, :])
                self.mem_dict[f"pos_{p}"] = sorted([self.knowledge_pairs[i][1] for i in idxs])

        elif self.tmpl_type == 'mc-pos':
            for c in range(self.num_chain_tmpl):
                for p in range(self.num_pos_tmpl):
                    idxs = np.flatnonzero(self.in_dist_mask[c, p, :])
                    self.mem_dict[str((c, p))] = sorted([self.knowledge_pairs[i][1] for i in idxs])

            for c in range(self.num_chain_tmpl):
                idxs = np.flatnonzero(self.in_dist_mask[c, :, :].any(axis=0))
                self.mem_dict[f"mc_{c}"] = sorted(list(set(self.knowledge_pairs[i][1] for i in idxs)))

            for p in range(self.num_pos_tmpl):
                idxs = np.flatnonzero(self.in_dist_mask[:, p, :].any(axis=0))
                self.mem_dict[f"pos_{p}"] = sorted(list(set(self.knowledge_pairs[i][1] for i in idxs)))

    def generate_in_dist_mask(self):

        if self.tmpl_type == 'mc-pos':
            self.generate_3dmask()

        elif self.tmpl_type == 'mc':
            num_tmpl = self.num_chain_tmpl 
            if self.ood_frac > -1:
                num_true_per_col = num_tmpl - int(num_tmpl * self.ood_frac)
                self.in_dist_mask, self.out_dist_mask = self.generate_2dmask(num_tmpl, self.M, num_true_per_col)
            else:
                self.in_dist_mask = np.ones((num_tmpl, self.M), dtype=bool)
                self.out_dist_mask = np.zeros((num_tmpl, self.M), dtype=bool)
            self.out_distV2_mask = None
            self.pos_chain_mask = None

        elif self.tmpl_type == 'pos':
            num_tmpl = self.num_pos_tmpl
            if self.ood_frac > -1:
                num_true_per_col = num_tmpl - int(num_tmpl * self.ood_frac)
                self.in_dist_mask, self.out_dist_mask = self.generate_2dmask(num_tmpl, self.M, num_true_per_col)
            else:
                self.in_dist_mask = np.ones((num_tmpl, self.M), dtype=bool)
                self.out_dist_mask = np.zeros((num_tmpl, self.M), dtype=bool)
            self.out_distV2_mask = None
            self.pos_chain_mask = None

        elif self.tmpl_type is None:
            self.in_dist_mask = None
            self.out_dist_mask = None
            self.out_distV2_mask = None
            self.pos_chain_mask = None

    def generate_2dmask(self, dim1, dim2, num_true_per_col):
        in_dist_mask = np.zeros((dim1, dim2), dtype=bool)

        # # Step 1: assign a distinct column to each row (requires num_tmpl ≤ M)
        #     # assert num_tmpl <= dim2, "Not enough columns to assign distinct entries per row"
        #     if dim1 <= dim2:
        #         initial_true_cols = np.random.permutation(dim2)[:dim1]
        #         in_dist_mask[np.arange(dim1), initial_true_cols] = True

        # Step 1: Ensure each row has at least one True
        for i in range(dim1):
            j = i % dim2  # wrap around if dim1 > dim2
            in_dist_mask[i, j] = True

        # Step 2: Fill in remaining True values per column
        for j in range(dim2):
            current_true_rows = np.flatnonzero(in_dist_mask[:, j])
            remaining = num_true_per_col - len(current_true_rows)
            if remaining > 0:
                eligible = np.flatnonzero(~in_dist_mask[:, j])
                chosen = np.random.choice(eligible, size=remaining, replace=False)
                in_dist_mask[chosen, j] = True

        out_dist_mask = ~in_dist_mask
        return in_dist_mask, out_dist_mask

    def generate_3dmask(self):
        if self.chain_per_pos > -1:
            self.pos_chain_mask, _ = self.generate_2dmask(self.num_chain_tmpl, self.num_pos_tmpl, self.chain_per_pos)


            num_true_per_kp = int(np.ceil((self.chain_per_pos * self.num_pos_tmpl) * (1 - self.ood_frac)))
            self.in_dist_mask = np.zeros((self.num_chain_tmpl, self.num_pos_tmpl, self.M), dtype=bool)
            
            # eligible = np.argwhere(self.pos_chain_mask)
            # for k in range(self.M):
            #     selected = eligible[np.random.choice(len(eligible), num_true_per_kp, replace=False)]
            #     for c, p in selected:
            #         self.in_dist_mask[c, p, k] = True

            eligible = np.argwhere(self.pos_chain_mask)
            
            # Step 1: Ensure every (chain, pos) has at least one kp
            for i, (c, p) in enumerate(eligible):
                k = i % self.M
                self.in_dist_mask[c, p, k] = True

            # Step 2: Fill in remaining trues per kp
            for k in range(self.M):
                current_true = np.argwhere(self.in_dist_mask[:, :, k])
                remaining = num_true_per_kp - len(current_true)
                if remaining > 0:
                    eligible_k = np.argwhere(self.pos_chain_mask & (~self.in_dist_mask[:, :, k]))
                    if len(eligible_k) >= remaining:
                        chosen = eligible_k[np.random.choice(len(eligible_k), size=remaining, replace=False)]
                        for c, p in chosen:
                            self.in_dist_mask[c, p, k] = True

            cp_mask_expanded = np.broadcast_to(self.pos_chain_mask[:, :, None], self.in_dist_mask.shape)
            self.out_distV2_mask = ~cp_mask_expanded  # shape (C, P, K), same False entries across all K
            self.out_dist_mask = cp_mask_expanded & (~self.in_dist_mask)

        else:
            self.pos_chain_mask = np.ones((self.num_chain_tmpl, self.num_pos_tmpl), dtype=bool)
            self.in_dist_mask = np.ones((self.num_chain_tmpl, self.num_pos_tmpl, self.M), dtype=bool)
            self.out_dist_mask = np.zeros((self.num_chain_tmpl, self.num_pos_tmpl, self.M), dtype=bool)
            self.out_distV2_mask = np.zeros((self.num_chain_tmpl, self.num_pos_tmpl, self.M), dtype=bool)

    def generate_knowledge_base(self):
        """
        Create M pairs of distinct tokens from range [0, V-1], so total 2M distinct tokens.
        Returns a list of pairs and the set of all tokens used in these pairs.
        """
        if self.V < 2 * self.M + 2: # so that we have at least two vocabs left for the MC
            err_str = f'choose a larger V or smaller M!'
            raise_and_log_error(err_str, err_type='value', logger=self.logger)

        # Sample 2M distinct tokens -- sample without replacement
        distinct_tokens = random.sample(range(self.V), 2*self.M)
        # Form them into M pairs
        knowledge_pairs = [
            (distinct_tokens[2*i], distinct_tokens[2*i+1])
            for i in range(self.M)
        ]
        return knowledge_pairs, sorted(distinct_tokens)
    
    def generate_position_templates(self, min_dist_mask=2, max_dist=None, first_half=True):

        """
        if first half is true: generate the templates such that the first element position is in the first half of the sequence
        """

        if self.bi_pos == "fixed":
            pos_tmpls = []
            valid_range = self.seq_length - 4 if (not first_half) else (self.seq_length // 2)
            distinct_positions = random.sample(range(valid_range), self.num_pos_tmpl)
            for i in range(self.num_pos_tmpl):
                pos_tmpls.append((distinct_positions[i], self.seq_length-1))

        elif self.bi_pos == "random":

            if 2 * self.num_pos_tmpl > self.seq_length - 2:
                err_str =  f'number of position templates [{self.num_pos_tmpl}] should be larger than (seq length - 2) / 2 [{(self.seq_length - 2) // 2}]'
                raise_and_log_error(err_str, err_type='value', logger=self.logger)

            # distinct_positions = random.sample(range(self.seq_length), 2*self.num_pos_tmpl)
            # pos_tmpls = [
            #     sorted((distinct_positions[2*i], distinct_positions[2*i+1]))
            #     for i in range(self.num_pos_tmpl)
            # ]

            if first_half:

                pos_tmpls = []
                
                distinct_positions_a = random.sample(range(self.seq_length // 2 - 1), self.num_pos_tmpl)
                distinct_positions_b = random.sample(range(self.seq_length // 2, self.seq_length-3), self.num_pos_tmpl)

                for a,b in zip(distinct_positions_a, distinct_positions_b):
                    pos_tmpls.append((a,b))

            else:


                num_trial = 5
                if max_dist is None:
                    max_dist = self.seq_length  # default upper bound if not provided


                for _ in range(num_trial):    
                    candidates = list(range(self.seq_length - 3))
                    # print('max pos possible: ', self.seq_length - 3)
                    random.shuffle(candidates)

                    used = set()
                    pos_tmpls = []

                    for i in range(len(candidates)):
                        a = candidates[i]
                        if a in used:
                            continue
                        for j in range(i + 1, len(candidates)):
                            b = candidates[j]
                            if b in used:
                                continue
                            dist = abs(a-b)
                            if min_dist_mask <= dist <= max_dist:
                                pos_tmpls.append(sorted((a, b)))
                                used.add(a)
                                used.add(b)
                                break
                        if len(pos_tmpls) == self.num_pos_tmpl:
                            break
                    
                    if len(pos_tmpls) == self.num_pos_tmpl:
                        break

            if len(pos_tmpls) < self.num_pos_tmpl:
                raise ValueError("Could not find enough valid pairs with the given constraints")

        return pos_tmpls

    def create_markov_transition_matrix(self, num_chain_tmpl=-1, uniform=False):
        """
        Creates transition matrix/matrices:
        - For order 1: standard Markov transition matrix
        - For order 0: rows are identical (unigram distribution)
        - If uniform=True: use uniform distribution across all tokens
        - Returns:
            - A single matrix if num_chain_tmpl == -1
            - A dictionary of matrices otherwise
        """
        if self.order > 2:
            raise ValueError('order > 2 not implemented')

        if uniform:
            return {0: np.ones((self.vocab_size_for_chain, self.vocab_size_for_chain)) / self.vocab_size_for_chain}
        
            
        matrices = {}

        
        # for i in range(max(num_chain_tmpl,1)):
        #     mat = np.random.rand(self.vocab_size_for_chain, self.vocab_size_for_chain)
        #     # normalize each row
        #     mat = mat / mat.sum(axis=1, keepdims=True)
        #     matrices[i] = mat
        
        # if num_chain_tmpl == -1: # for the online case, return the matrix directly, not in the form of dictionary...
        #     return mat
        # else:
        #     return matrices
        
        alpha = np.ones(self.vocab_size_for_chain)  # Dirichlet parameter vector [1, ..., 1]
        matrices = {}
        num_rows = self.vocab_size_for_chain ** self.order  # number of contexts


        for i in range(max(num_chain_tmpl, 1)):
            if self.order >= 1:
                mat = np.array([
                    np.random.dirichlet(alpha) for _ in range(num_rows)
                ])
                matrices[i] = mat

            elif self.order == 0:
                # Same row repeated — represents a unigram model
                unigram = np.random.dirichlet(alpha)
                mat = np.tile(unigram[None, :], (self.vocab_size_for_chain, 1))
                matrices[i] = mat

        return matrices if num_chain_tmpl != -1 else mat
   
    def get_empirical_stationary_context_dist(self, transition_matrix, k, num_samples=1000):
        """
        Simulate a long sequence and collect empirical stationary distribution over order-k contexts.
        """
        V = self.vocab_size_for_chain
        seq = self.sample_markov_chain(transition_matrix, num_samples + k, start_mode="uniform")
        counts = np.zeros(V ** k)
        for i in range(num_samples):
            ctx = seq[i:i + k]
            idx = sum([ctx[j] * (V ** (k - j - 1)) for j in range(k)])
            counts[idx] += 1
        probs = counts / counts.sum()
        return probs
    
    def sample_markov_chain(self, transition_matrix: np.ndarray, length: int, start_mode=None):
        """
        Given the transition matrix and a desired length, sample a sequence.
        """

        if start_mode is None:
            start_mode = self.initial

        # initial distribution
        if start_mode == "stationary":
            if self.order <= 1:
                # Compute stationary distribution from left eigenvector of transition matrix
                eigvals, eigvecs = np.linalg.eig(transition_matrix.T)
                stationary = np.real(eigvecs[:, np.isclose(eigvals, 1)])
                stationary = stationary[:, 0]
                stationary = stationary / stationary.sum()
                current_token = np.random.choice(self.vocab_size_for_chain, p=stationary)
                context = [current_token]
            else:
                # probs = self.get_empirical_stationary_context_dist(transition_matrix, self.order)
                # init_idx = np.random.choice(self.vocab_size_for_chain ** self.order, p=probs)
                # context = []
                # for i in range(self.order):
                #     context.append(init_idx // (self.vocab_size_for_chain ** (self.order - i - 1)))
                #     init_idx %= (self.vocab_size_for_chain ** (self.order - i - 1))
                context = [np.random.choice(self.vocab_size_for_chain) for _ in range(self.order)]
                # TODO: it's not statioanry -- i think this is making it super slow...
        else:
            # Uniform initialization
            # current_token = np.random.choice(self.vocab_size_for_chain)
            context = [np.random.choice(self.vocab_size_for_chain) for _ in range(self.order)]

        # sequence = [current_token]
        # for _ in range(length - 1):
        #     probs = transition_matrix[current_token]
        #     current_token = np.random.choice(self.vocab_size_for_chain, p=probs)
        #     sequence.append(current_token)
        sequence = context[:]
        for _ in range(length - self.order):
            ctx_idx = sum([context[i] * (self.vocab_size_for_chain ** (self.order - i - 1)) for i in range(self.order)])
            next_token = np.random.choice(self.vocab_size_for_chain, p=transition_matrix[ctx_idx])
            sequence.append(next_token)
            context = context[1:] + [next_token]
        return sequence

    def generate_data(self, num_sequences: int, only_ai: bool = False, skip_kb: bool = False, ood: bool = False, ood_struct: bool = False):
        """
        Generates a dataset of sequences with the knowledge base inserted.

        Args:
            num_sequences (int): Number of sequences to generate.
            skip_kb: skip replacing knowledge tokens for debugging purposes -- temporary, remove --

        Returns:
            A torch.LongTensor of shape [num_sequences, seq_length] (or variable length if not fixed)
            and the knowledge base pairs.
        """

        kp_list = random.choices(self.knowledge_pairs, k=num_sequences) if self.M > 0 else []

        # --------------- choose and define templates (mc,pos) 
        # if no template exist (-1,-1)
        
        if self.tmpl_type == 'mc-pos':
            if ood and ood_struct:
                raise_and_log_error("choose one type of ood!", err_type='value', logger=self.logger)
            
            kp_tmpl_mask = self.in_dist_mask if ((not ood) and (not ood_struct)) else self.out_dist_mask if ood else self.out_distV2_mask
            # print(np.sum(np.array(kp_tmpl_mask) == np.array(self.in_dist_mask)))
            # print(np.array(kp_tmpl_mask))
            # sys.exit(0)
            
        else:
            kp_tmpl_mask = self.out_dist_mask if ood else self.in_dist_mask
        if ((ood) or (ood_struct)) and (not self.use_ood):
            raise_and_log_error('ood is not valid', err_type='value', logger=self.logger)
        
        if self.tmpl_type is None:
            tmpl_list = [(-1,-1)] * num_sequences

 
        elif self.tmpl_type == 'mc':
            true_indices = [np.flatnonzero(kp_tmpl_mask[:, self.kp_dict[col]]) for col in kp_list]
            mc_list = [arr[np.random.randint(len(arr))] for arr in true_indices if len(arr) > 0]
            kp_list = [kp for kp, arr in zip(kp_list, true_indices) if len(arr) > 0]
            tmpl_list = [(id, -1) for id in mc_list]

        elif self.tmpl_type == 'pos':
            true_indices = [np.flatnonzero(kp_tmpl_mask[:, self.kp_dict[col]]) for col in kp_list]
            pos_list = [arr[np.random.randint(len(arr))] for arr in true_indices if len(arr) > 0]
            kp_list = [kp for kp, arr in zip(kp_list, true_indices) if len(arr) > 0]
            tmpl_list = [(-1, id) for id in pos_list]

        elif self.tmpl_type == 'mc-pos':
            kp_masked = []
            tmpl_list = []
            for col in kp_list:
                valid_indices = np.argwhere(kp_tmpl_mask[:, :, self.kp_dict[col]])
                if len(valid_indices) > 0:
                    sampled = valid_indices[np.random.randint(len(valid_indices))]
                    tmpl_list.append((sampled[0], sampled[1]))
                    kp_masked.append(col)

                # print(f'kp:{col}, valid idx {valid_indices}, sampled: {sampled}')
            kp_list = kp_masked
            # sys.exit(0)

        # data = {'chain':[], 'kp':kp_list, 'tmpl':tmpl_list}
        data = {'chain':[], 'kp':kp_list, 'tmpl':tmpl_list, 'transition_matrix':[]}

        if len(kp_list) < num_sequences:
            self.logger.warning(f'size of the final dataset {len(kp_list)} smaller than {num_sequences}!!!!!!')
            num_sequences = len(kp_list)

        for idx in range(num_sequences):
            
            length = self.seq_length 
            
            tmpl_chain, tmpl_pos = tmpl_list[idx]
            kp = kp_list[idx] if self.M > 0 else None

            if only_ai:
                length = length // 2 + 1

            # --------------- sequence generation
            if tmpl_chain == -1:
                transition_matrix = self.create_markov_transition_matrix(num_chain_tmpl=-1) # along axis=1 sums to 1
            else:
                transition_matrix = self.transition_mat_dict[tmpl_chain]

            chain_seq = self.sample_markov_chain(transition_matrix, length)
            mapped_seq = [self.chain_tokens[token] for token in chain_seq]

            data['transition_matrix'].append(transition_matrix)
            # --------------- insertion of knowledge
            if (not skip_kb) and self.M > 0: 
                if tmpl_pos != -1:
                    pos1, pos2 = self.pos_tmpls[tmpl_pos]
                    if only_ai:
                        pos2 = None
                else:
                    if only_ai:
                        pos1 = random.sample(range(length-4), 1)[0]
                        pos2 = None
                    else:
                        if self.bi_pos == "fixed":
                            pos1 = random.sample(range(length-4), 1)[0]
                            pos2 = length - 1
                        elif self.bi_pos == "random":
                            pos1, pos2 = sorted(random.sample(range(length - 3), 2))
                        
                if self.insertion_mode == "replace":
                    mapped_seq[pos1] = kp[0]
                    if pos2 is not None:
                        mapped_seq[pos2] = kp[1]
                elif self.insertion_mode == "shift":
                    if pos2 is None:
                        markov_content = mapped_seq[:length - 1]  # trim 1 token
                        mapped_seq = markov_content[:pos1] + [kp[0]] + markov_content[pos1:]
                    else:
                        markov_content = mapped_seq[:length-2]
                        mapped_seq =  markov_content[:pos1] + [kp[0]] + markov_content[pos1:pos2-1] + [kp[1]] + markov_content[pos2-1:]
                else:
                    raise ValueError('insertion mode invalid')
            
            data["chain"].append(mapped_seq)

        return data

    def generate_batch(self, num_sequences, max_length, mode='window', stride=None, dtype=torch.long, only_ai: bool = False, skip_kb: bool = False, ood: bool = False, ood_struct: bool = False):

        """
        adds special tokens to the sequences, make all the sequence lengths equal by padding, truncate sequences if long
        """

        data = self.generate_data(num_sequences, only_ai=only_ai, skip_kb=skip_kb, ood=ood, ood_struct=ood_struct)
        sequences, kps, tmpls = data['chain'], data['kp'], data['tmpl']

        if self.special_toks is not None:
            if not self.skip_spectok:
                self.BOS = self.special_toks['BOS']
                self.EOS = self.special_toks['EOS']
                self.PAD = self.special_toks['PAD']


        # padding and truncations -- input and target assignment
        inputs = []
        targets = []
        for seq in sequences:
            # # --------- EOS, BOS, PAD, UNK -------------------------
            # seq_extd = [self.BOS] + seq + [self.EOS]
            seq_extd = seq[:-1] if self.skip_spectok else [self.BOS] + seq 

            # ----------- fitting into context window ------------------
            if len(seq_extd) <= max_length:
                input_seq = seq_extd
                # target_seq = seq_extd[1:max_length] + [self.PAD]
                target_seq = seq[1:max_length] if self.skip_spectok else seq_extd[1:] + [self.EOS]
                inputs.append(input_seq)
                targets.append(target_seq)
            else:
                if mode == 'truncate':
                    input_seq = seq_extd[:max_length]         # we could also add a TRUNC token for to flag when it's truncated, but not needed now.
                    target_seq = seq_extd[1:max_length + 1]
                    inputs.append(input_seq)
                    targets.append(target_seq)
                elif mode == 'split':
                    for i in range(0, len(seq_extd), max_length):
                        input_seq = seq_extd[i:i+max_length]  
                        target_seq = seq_extd[i + 1:i + max_length + 1]
                        if len(target_seq) < len(input_seq):
                            print(len(target_seq) < len(input_seq))
                            # target_seq = target_seq + [self.PAD]
                            target_seq = target_seq + [self.PAD] if self.skip_spectok else target_seq + [self.EOS]
                        inputs.append(input_seq)
                        targets.append(target_seq)
                elif mode == 'window':
                    for i in range(0, len(seq_extd) - max_length + 1, stride):
                        input_seq = seq_extd[i:i + max_length]
                        target_seq = seq_extd[i + 1:i + max_length + 1]
                        if len(target_seq) < len(input_seq):
                            print(len(target_seq) < len(input_seq))
                            # target_seq = target_seq + [self.PAD]
                            target_seq = target_seq + [self.PAD] if self.skip_spectok else target_seq + [self.EOS]
                            
                        inputs.append(input_seq)
                        targets.append(target_seq)
                else:
                    raise ValueError("Mode should be 'truncate', 'split', or 'window'.")
                
        inputs = torch.tensor(inputs, dtype=dtype)
        targets = torch.tensor(targets, dtype=dtype)

        if not self.skip_spectok:
            pad_idx = self.PAD
            padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_idx)  # dim: B x T 
            padded_targets = pad_sequence(targets, batch_first=True, padding_value=pad_idx)  # dim: B x T 
            # if eos_idx is None:
            attn_masks = (padded_inputs != pad_idx).long()
        else:
            padded_inputs = inputs
            padded_targets = targets
            attn_masks = torch.ones_like(padded_inputs, dtype=torch.long)

        batch = {
                'x': padded_inputs,
                'y': padded_targets,
                'attn_mask': attn_masks,
                'kps': kps,
                'tmpls': tmpls,
                'transition_matrix': data['transition_matrix']
                }

        return batch

    def get_knowledge_base(self):
        return self.knowledge_pairs, self.knowledge_tokens
    
    def get_MC_props(self):
        return self.chain_tokens, self.num_chain_tmpl
    
    def get_transition_matrix(self, idx=None):
        if idx is not None:
            return self.transition_mat_dict[idx]
        else: 
            return self.transition_mat_dict
        
    
