import torch
# import torch_npu
from transformers.generation.logits_process import LogitsProcessor


class BinaryOutputProcessor(LogitsProcessor):
    def __init__(self, tokenizer, true_token_id, false_token_id, eos_token_id):
        self.true_token_id = true_token_id
        self.false_token_id = false_token_id
        self.eos_token_id = eos_token_id
        self.allowed_tokens = [true_token_id, false_token_id]
        self.generated = False  # 跟踪是否已生成True/False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.generated:
            # 第一步：只允许True或False
            mask = torch.full_like(scores, float('-inf'))
            mask[:, self.allowed_tokens] = 0
            self.generated = True
            return scores + mask
        else:
            # 生成后：强制EOS停止
            mask = torch.full_like(scores, float('-inf'))
            mask[:, self.eos_token_id] = 0
            self.generated = False
            return scores + mask
