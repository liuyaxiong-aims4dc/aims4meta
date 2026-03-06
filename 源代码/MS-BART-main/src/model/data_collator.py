import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from numpy.random import permutation, poisson
from transformers.tokenization_utils_base import  PreTrainedTokenizerBase

# https://github.com/morganmcg1/rotobart/blob/cd29e458d0adfa270d47e11614427935541aef0c/data_collator.py#L223

@dataclass
class DataCollatorForDenoisingTasks:
    """Data collator used denoising language modeling task in BART.
    The implementation is based on
    https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/fairseq/data/denoising_dataset.py.
    The default paramters is based on BART paper https://arxiv.org/abs/1910.13461.
    """

    tokenizer: PreTrainedTokenizerBase
    mask_ratio: float = 0.3
    poisson_lambda: float = 3.0
    permutate_sentence_ratio: float = 0.0    # BART 有排列去噪任务，我们这里不需要
    pad_to_multiple_of: int = 16

    def __post_init__(self):
        if self.tokenizer.mask_token is None or self.tokenizer.eos_token is None:
            raise ValueError

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, np.ndarray]:
        """Batching, adding whole word mask and permutate sentences
        Args:
            examples (dict): list of examples each examples contains input_ids field
        """
        # Handle dict or lists with proper padding and conversion to tensor.
        final_input_ids = []
        final_labels = []
        final_decoder_input_ids = []

        max_input_len = max(len(inputs["input_ids"]) for inputs in examples)
        raw_input_ids = np.array([
            np.pad(
                inputs["input_ids"], 
                pad_width=(0, max_input_len - len(inputs["input_ids"])),  # 在末尾填充 (pad_right = max_len - len)
                mode='constant', 
                constant_values=self.tokenizer.pad_token_id
            ) 
            for inputs in examples
        ])

        max_len = max(len(label["labels"]) for label in examples)
        raw_labels = np.array([
            np.pad(
                label["labels"], 
                pad_width=(0, max_len - len(label["labels"])),  # 在末尾填充 (pad_right = max_len - len)
                mode='constant', 
                constant_values=-100
            )
            for label in examples
        ])
        data_class = [e["data_class"] for e in examples]
        examples_cnt = len(data_class)
        assert len(raw_input_ids) == len(raw_labels) == examples_cnt, "DataCollatorForDenoisingTasks: input_ids and labels should have the same length"

        final_labels = raw_labels.copy()
        final_decoder_input_ids = self.shift_tokens_right(raw_labels)

        sep_token_id = self.tokenizer.convert_tokens_to_ids("<fps_sep>")
        for i in range(examples_cnt):
            cur_data_class = data_class[i]
            cur_input_ids = raw_input_ids[i]
            if cur_data_class == 1:
                indices = np.where(cur_input_ids == sep_token_id)[0]
                if indices.size > 0:
                    sep_id_index = indices[0]
                else:
                    raise ValueError("max seq length is too small")
                selfies_sep_fps = False
                if cur_input_ids[sep_id_index+1] > sep_token_id:
                    # 说明 <fps>在<sep>后面,数据格式为 <selfies><sep><fps>
                    selfies_input_ids = cur_input_ids[:sep_id_index]
                    fps_input_ids = cur_input_ids[sep_id_index+1:]
                    selfies_sep_fps = True
                else:
                    # 说明 <fps>在<sep>前面,数据格式为 <fps><sep><selfies>
                    selfies_input_ids = cur_input_ids[sep_id_index+1:]
                    fps_input_ids = cur_input_ids[:sep_id_index]
                selfies_input_ids_mask, _ = self.add_whole_word_mask(np.array([selfies_input_ids]), False)
                if selfies_sep_fps:
                    cur_input_ids = np.concatenate([selfies_input_ids_mask[0], [sep_token_id], fps_input_ids])
                else:
                    cur_input_ids = np.concatenate([fps_input_ids, [sep_token_id], selfies_input_ids_mask[0]])
                final_input_ids.append(cur_input_ids)
            elif cur_data_class == 2:
                cur_input_ids, _ = self.add_whole_word_mask(np.array([cur_input_ids]), False)
                final_input_ids.append(cur_input_ids[0])
            elif cur_data_class == 3:
                final_input_ids.append(cur_input_ids)
        
        final_decoder_input_ids = np.array(final_decoder_input_ids)
        final_decoder_input_ids[final_decoder_input_ids == -100] = self.tokenizer.pad_token_id
        
        batch =  {
            "input_ids": torch.LongTensor(np.array(final_input_ids)),
            "decoder_input_ids": torch.LongTensor(final_decoder_input_ids),
            "labels": torch.LongTensor(np.array(final_labels)),
        }
        return batch

    def shift_tokens_right(self, inputs):
        """Shift decoder input ids right: https://github.com/huggingface/transformers/issues/7961.
        Examples:
            <s>My dog is cute.</s><s>It loves to play in the park.</s><pad><pad>
            shift to -> </s><s>My dog is cute.</s><s>It loves to play in the park.<pad><pad>
        """

        shifted_inputs = np.roll(inputs, 1, axis=-1)

        # replace first token with eos token
        shifted_inputs[:, 0] = self.tokenizer.eos_token_id

        # when there's padding, the last eos tokens will not be rotate to first positon
        # we'll need to replace it with a padding token

        # replace eos tokens at the end of sequences with pad tokens
        end_with_eos = np.where(shifted_inputs[:, -1] == self.tokenizer.eos_token_id)
        shifted_inputs[end_with_eos, -1] = self.tokenizer.pad_token_id

        # find positions where where's the token is eos and its follwing token is a padding token
        last_eos_indices = np.where(
            (shifted_inputs[:, :-1] == self.tokenizer.eos_token_id)
            * (shifted_inputs[:, 1:] == self.tokenizer.pad_token_id)
        )

        # replace eos tokens with pad token
        shifted_inputs[last_eos_indices] = self.tokenizer.pad_token_id
        return shifted_inputs

    def permutate_sentences(self, inputs):
        results = inputs.copy()

        full_stops = inputs == self.tokenizer.eos_token_id

        sentence_ends = np.argwhere(full_stops[:, 1:] * ~full_stops[:, :-1])
        sentence_ends[:, 1] += 2
        num_sentences = np.unique(sentence_ends[:, 0], return_counts=True)[1]
        num_to_permute = np.ceil((num_sentences * 2 * self.permutate_sentence_ratio) / 2.0).astype(int)

        sentence_ends = np.split(sentence_ends[:, 1], np.unique(sentence_ends[:, 0], return_index=True)[1][1:])

        for i in range(inputs.shape[0]):
            substitutions = np.random.permutation(num_sentences[i])[: num_to_permute[i]]

            ordering = np.arange(0, num_sentences[i])
            ordering[substitutions] = substitutions[np.random.permutation(num_to_permute[i])]

            index = 0
            for j in ordering:
                sentence = inputs[i, (sentence_ends[i][j - 1] if j > 0 else 0) : sentence_ends[i][j]]
                results[i, index : index + sentence.shape[0]] = sentence
                index += sentence.shape[0]
        return results

    def add_whole_word_mask(self, inputs, do_permutate):
        labels = inputs.copy()

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = np.array(special_tokens_mask, dtype=bool)

        # determine how many tokens we need to mask in total
        is_token = ~(labels == self.tokenizer.pad_token_id) & ~special_tokens_mask
        num_to_mask = int(math.ceil(is_token.astype(float).sum() * self.mask_ratio))
        if num_to_mask == 0:
            return inputs, labels

        # generate a sufficient number of span lengths
        lengths = poisson(lam=self.poisson_lambda, size=(num_to_mask,))
        while np.cumsum(lengths, 0)[-1] < num_to_mask:
            lengths = np.concatenate([lengths, poisson(lam=self.poisson_lambda, size=(num_to_mask,))])

        # remove all spans of length 0
        # Note that BART inserts additional mask tokens where length == 0,
        # which we do not implement for now as it adds additional complexity
        lengths = lengths[lengths > 0]

        # trim to about num_to_mask tokens
        idx = np.argmin(np.abs(np.cumsum(lengths, 0) - num_to_mask)) + 1
        lengths = lengths[: idx + 1]

        # select span start indices
        # print("IS TOKEN")
        # print(is_token)
        # print(sum(list(map(lambda x: 1 if(x) else 0, is_token[0]))))
        token_indices = np.argwhere(is_token == 1)
        # print("TOKEN INDICES")
        # print(token_indices)
        span_starts = permutation(token_indices.shape[0])[: lengths.shape[0]]

        # prepare mask
        masked_indices = np.array(token_indices[span_starts])
        # print("MASKED INDICES")
        # print(masked_indices)
        mask = np.full_like(labels, fill_value=False)

        # mask span start indices
        for mi in masked_indices:
            mask[tuple(mi)] = True
        lengths -= 1

        # fill up spans
        max_index = labels.shape[1] - 1
        remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)
        while np.any(remaining):
            masked_indices[remaining, 1] += 1
            for mi in masked_indices:
                mask[tuple(mi)] = True
            lengths -= 1
            remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)

        # place the mask tokens
        mask[np.where(special_tokens_mask)] = False
        inputs[np.where(mask)] = self.tokenizer.mask_token_id

        if not do_permutate:
            labels[np.where(mask)] = -100
        else:
            labels[np.where(special_tokens_mask)] = -100

        # remove mask tokens that are not starts of spans
        to_remove = (mask == 1) & np.roll((mask == 1), 1, 1)
        new_inputs = np.full_like(labels, fill_value=self.tokenizer.pad_token_id)

        # splits = list(map(lambda x: x.reshape(-1),  np.split(inputs_copy, indices_or_sections=2, axis=0))
        for i, example in enumerate(np.split(inputs, indices_or_sections=new_inputs.shape[0], axis=0)):
            new_example = example[0][~to_remove[i]]
            new_inputs[i, 0 : new_example.shape[0]] = new_example

        # batching now fixed
        return new_inputs, labels
    
@dataclass
class FormulaDataCollatorForDenoisingTasks:
    """Data collator used denoising language modeling task in BART.
    The implementation is based on
    https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/fairseq/data/denoising_dataset.py.
    The default paramters is based on BART paper https://arxiv.org/abs/1910.13461.
    """

    tokenizer: PreTrainedTokenizerBase
    mask_ratio: float = 0.3
    poisson_lambda: float = 3.0
    permutate_sentence_ratio: float = 0.0    # BART 有排列去噪任务，我们这里不需要
    pad_to_multiple_of: int = 16

    def __post_init__(self):
        if self.tokenizer.mask_token is None or self.tokenizer.eos_token is None:
            raise ValueError

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, np.ndarray]:
        """Batching, adding whole word mask and permutate sentences
        Args:
            examples (dict): list of examples each examples contains input_ids field
        """
        # Handle dict or lists with proper padding and conversion to tensor.
        final_input_ids = []
        final_labels = []
        final_decoder_input_ids = []

        max_input_len = max(len(inputs["input_ids"]) for inputs in examples)
        raw_input_ids = np.array([
            np.pad(
                inputs["input_ids"], 
                pad_width=(0, max_input_len - len(inputs["input_ids"])),  # 在末尾填充 (pad_right = max_len - len)
                mode='constant', 
                constant_values=self.tokenizer.pad_token_id
            ) 
            for inputs in examples
        ])

        max_len = max(len(label["labels"]) for label in examples)
        raw_labels = np.array([
            np.pad(
                label["labels"], 
                pad_width=(0, max_len - len(label["labels"])),  # 在末尾填充 (pad_right = max_len - len)
                mode='constant', 
                constant_values=-100
            )
            for label in examples
        ])
        data_class = [e["data_class"] for e in examples]
        examples_cnt = len(data_class)
        assert len(raw_input_ids) == len(raw_labels) == examples_cnt, "DataCollatorForDenoisingTasks: input_ids and labels should have the same length"

        final_labels = raw_labels.copy()
        final_decoder_input_ids = self.shift_tokens_right(raw_labels)

        sep_token_id = self.tokenizer.convert_tokens_to_ids("<fps_sep>")
        formula_sep_token_id = self.tokenizer.convert_tokens_to_ids("<formula_sep>")
        for i in range(examples_cnt):
            cur_data_class = data_class[i]
            cur_input_ids = raw_input_ids[i]
            if cur_data_class == 1:
                sep_token_id_idx = np.where(cur_input_ids == sep_token_id)[0]
                formula_sep_token_id_idx = np.where(cur_input_ids == formula_sep_token_id)[0]
                # Check if tokens are found
                if sep_token_id_idx.size == 0 or formula_sep_token_id_idx.size == 0:
                    raise ValueError("Separator tokens not found in input_ids")
                sep_token_id_idx = sep_token_id_idx[0]
                formula_sep_token_id_idx = formula_sep_token_id_idx[0]

                # Initialize output
                formula_id = []
                fps_id = []
                selfies_input_ids = []

                if sep_token_id_idx > formula_sep_token_id_idx:
                    # Structure: formula<formula_sep_token>fps<sep_token>selfies
                    formula_id = cur_input_ids[:formula_sep_token_id_idx]
                    fps_id = cur_input_ids[formula_sep_token_id_idx + 1:sep_token_id_idx]
                    selfies_input_ids = cur_input_ids[sep_token_id_idx + 1:]
                    selfies_input_ids_mask, _ = self.add_whole_word_mask(np.array([selfies_input_ids]), False)
                    cur_input_ids = np.concatenate([formula_id, [formula_sep_token_id], fps_id, [sep_token_id], selfies_input_ids_mask[0]])
                else:
                    # Structure: selfies<sep_token>fps<formula_sep_token>formula
                    selfies_input_ids = cur_input_ids[:sep_token_id_idx]
                    fps_id = cur_input_ids[sep_token_id_idx + 1:formula_sep_token_id_idx]
                    formula_id = cur_input_ids[formula_sep_token_id_idx + 1:]
                    selfies_input_ids_mask, _ = self.add_whole_word_mask(np.array([selfies_input_ids]), False)
                    cur_input_ids = np.concatenate([selfies_input_ids_mask[0], [sep_token_id], fps_id, [formula_sep_token_id], formula_id])
                final_input_ids.append(cur_input_ids)
            elif cur_data_class == 2:
                # Find index of formula_sep_token_id
                formula_sep_idx = np.where(cur_input_ids == formula_sep_token_id)[0]
                if formula_sep_idx.size == 0:
                    raise ValueError("formula_sep_token_id not found in cur_input_ids")
                formula_sep_idx = formula_sep_idx[0]
                # Split cur_input_ids into formula_id and selfies_input_ids
                formula_id = cur_input_ids[:formula_sep_idx]
                selfies_input_ids = cur_input_ids[formula_sep_idx + 1:]
                selfies_input_ids_mask, _ = self.add_whole_word_mask(np.array([selfies_input_ids]), False)
                cur_input_ids = np.concatenate([formula_id, [formula_sep_token_id], selfies_input_ids_mask[0]])
                final_input_ids.append(cur_input_ids)
            elif cur_data_class == 3:
                final_input_ids.append(cur_input_ids)
        
        final_decoder_input_ids = np.array(final_decoder_input_ids)
        final_decoder_input_ids[final_decoder_input_ids == -100] = self.tokenizer.pad_token_id
        
        batch =  {
            "input_ids": torch.LongTensor(np.array(final_input_ids)),
            "decoder_input_ids": torch.LongTensor(final_decoder_input_ids),
            "labels": torch.LongTensor(np.array(final_labels)),
        }
        return batch

    def shift_tokens_right(self, inputs):
        """Shift decoder input ids right: https://github.com/huggingface/transformers/issues/7961.
        Examples:
            <s>My dog is cute.</s><s>It loves to play in the park.</s><pad><pad>
            shift to -> </s><s>My dog is cute.</s><s>It loves to play in the park.<pad><pad>
        """

        shifted_inputs = np.roll(inputs, 1, axis=-1)

        # replace first token with eos token
        shifted_inputs[:, 0] = self.tokenizer.eos_token_id

        # when there's padding, the last eos tokens will not be rotate to first positon
        # we'll need to replace it with a padding token

        # replace eos tokens at the end of sequences with pad tokens
        end_with_eos = np.where(shifted_inputs[:, -1] == self.tokenizer.eos_token_id)
        shifted_inputs[end_with_eos, -1] = self.tokenizer.pad_token_id

        # find positions where where's the token is eos and its follwing token is a padding token
        last_eos_indices = np.where(
            (shifted_inputs[:, :-1] == self.tokenizer.eos_token_id)
            * (shifted_inputs[:, 1:] == self.tokenizer.pad_token_id)
        )

        # replace eos tokens with pad token
        shifted_inputs[last_eos_indices] = self.tokenizer.pad_token_id
        return shifted_inputs

    def permutate_sentences(self, inputs):
        results = inputs.copy()

        full_stops = inputs == self.tokenizer.eos_token_id

        sentence_ends = np.argwhere(full_stops[:, 1:] * ~full_stops[:, :-1])
        sentence_ends[:, 1] += 2
        num_sentences = np.unique(sentence_ends[:, 0], return_counts=True)[1]
        num_to_permute = np.ceil((num_sentences * 2 * self.permutate_sentence_ratio) / 2.0).astype(int)

        sentence_ends = np.split(sentence_ends[:, 1], np.unique(sentence_ends[:, 0], return_index=True)[1][1:])

        for i in range(inputs.shape[0]):
            substitutions = np.random.permutation(num_sentences[i])[: num_to_permute[i]]

            ordering = np.arange(0, num_sentences[i])
            ordering[substitutions] = substitutions[np.random.permutation(num_to_permute[i])]

            index = 0
            for j in ordering:
                sentence = inputs[i, (sentence_ends[i][j - 1] if j > 0 else 0) : sentence_ends[i][j]]
                results[i, index : index + sentence.shape[0]] = sentence
                index += sentence.shape[0]
        return results

    def add_whole_word_mask(self, inputs, do_permutate):
        labels = inputs.copy()

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = np.array(special_tokens_mask, dtype=bool)

        # determine how many tokens we need to mask in total
        is_token = ~(labels == self.tokenizer.pad_token_id) & ~special_tokens_mask
        num_to_mask = int(math.ceil(is_token.astype(float).sum() * self.mask_ratio))
        if num_to_mask == 0:
            return inputs, labels

        # generate a sufficient number of span lengths
        lengths = poisson(lam=self.poisson_lambda, size=(num_to_mask,))
        while np.cumsum(lengths, 0)[-1] < num_to_mask:
            lengths = np.concatenate([lengths, poisson(lam=self.poisson_lambda, size=(num_to_mask,))])

        # remove all spans of length 0
        # Note that BART inserts additional mask tokens where length == 0,
        # which we do not implement for now as it adds additional complexity
        lengths = lengths[lengths > 0]

        # trim to about num_to_mask tokens
        idx = np.argmin(np.abs(np.cumsum(lengths, 0) - num_to_mask)) + 1
        lengths = lengths[: idx + 1]

        # select span start indices
        # print("IS TOKEN")
        # print(is_token)
        # print(sum(list(map(lambda x: 1 if(x) else 0, is_token[0]))))
        token_indices = np.argwhere(is_token == 1)
        # print("TOKEN INDICES")
        # print(token_indices)
        span_starts = permutation(token_indices.shape[0])[: lengths.shape[0]]

        # prepare mask
        masked_indices = np.array(token_indices[span_starts])
        # print("MASKED INDICES")
        # print(masked_indices)
        mask = np.full_like(labels, fill_value=False)

        # mask span start indices
        for mi in masked_indices:
            mask[tuple(mi)] = True
        lengths -= 1

        # fill up spans
        max_index = labels.shape[1] - 1
        remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)
        while np.any(remaining):
            masked_indices[remaining, 1] += 1
            for mi in masked_indices:
                mask[tuple(mi)] = True
            lengths -= 1
            remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)

        # place the mask tokens
        mask[np.where(special_tokens_mask)] = False
        inputs[np.where(mask)] = self.tokenizer.mask_token_id

        if not do_permutate:
            labels[np.where(mask)] = -100
        else:
            labels[np.where(special_tokens_mask)] = -100

        # remove mask tokens that are not starts of spans
        to_remove = (mask == 1) & np.roll((mask == 1), 1, 1)
        new_inputs = np.full_like(labels, fill_value=self.tokenizer.pad_token_id)

        # splits = list(map(lambda x: x.reshape(-1),  np.split(inputs_copy, indices_or_sections=2, axis=0))
        for i, example in enumerate(np.split(inputs, indices_or_sections=new_inputs.shape[0], axis=0)):
            new_example = example[0][~to_remove[i]]
            new_inputs[i, 0 : new_example.shape[0]] = new_example

        # batching now fixed
        return new_inputs, labels