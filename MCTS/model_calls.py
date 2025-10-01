import os
import time
import math
from typing import List, Tuple, Optional
from tqdm.auto import tqdm
import torch.distributed as dist
from utils import is_distributed, ddp_setup, shard_indices
from setup_logger import setup_logger
from logits_processor import BinaryOutputProcessor
from transformers.generation.logits_process import LogitsProcessorList
import torch
# import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalLLM:
    def __init__(
        self, llm_path: str, dtype: str, output_folder: str, no_sample: bool = True
    ):
        self.logger = setup_logger(self.__class__.__name__)
        self.llm_path = llm_path
        self.dtype = dtype
        self.no_sample = no_sample
        self.output_folder = output_folder

        self.setup_device()
        self.load_model()

    def process_data(
        self,
        prepared_data: list,
        batch_size: int,
        max_new_tokens: int,
        binary_output: bool = False,
    ) -> List[Tuple[int, str]]:
        """
        ä½¿ç”¨å¤§æ¨¡åž‹å¯¹å¤–éƒ¨ä¼ å…¥çš„ prompt åˆ—è¡¨é€�ä¸€ç”Ÿæˆ�è¾“å‡ºï¼Œæ”¯æŒ�åˆ†å¸ƒå¼�ã€‚
        è¿”å›ž [(idx, output_str), ...]ï¼Œå…¶ä¸­ idx æ˜¯è¾“å…¥çš„ç´¢å¼•ï¼Œæ–¹ä¾¿è¿˜åŽŸé¡ºåº�ã€‚
        """
        # åˆ†å¸ƒå¼�æ•°æ�®åˆ‡åˆ†
        indices = shard_indices(len(prepared_data), self.rank, self.world_size)
        local_data = [(i, prepared_data[i]) for i in indices]

        if self.rank == 0:
            self.logger.info(
                f"Processing {len(prepared_data)} prompts across {self.world_size} processes"
            )
            self.logger.info(
                f"Local process {self.rank} handling {len(local_data)} prompts"
            )

        if binary_output:
            true_id = self.tokenizer.convert_tokens_to_ids("True")
            false_id = self.tokenizer.convert_tokens_to_ids("False")
            eos_id = self.tokenizer.eos_token_id
            processor = BinaryOutputProcessor(
                self.tokenizer, true_id, false_id, eos_id)
            logits_processor = LogitsProcessorList([processor])

        local_outputs = []
        progress = tqdm(
            total=len(local_data),
            desc=f"Processing (rank {self.rank})",
            disable=(self.rank != 0),
        )

        # å¤„ç�†æ•°æ�®
        for i in range(0, len(local_data), batch_size):
            batch_items = local_data[i: i + batch_size]
            batch_idx, batch_prompts = zip(*batch_items)

            inputs = self.tokenizer(
                list(batch_prompts), return_tensors="pt", padding=True, truncation=True
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            tok_embeds = self.llm_model.get_input_embeddings()(input_ids)

            if not binary_output:
                gen_ids = self.llm_model.generate(
                    inputs_embeds=tok_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=not self.no_sample,
                    use_cache=True,
                    logits_processor=None,
                )
            else:
                gen_ids = self.llm_model.generate(
                    inputs_embeds=tok_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=5,
                    do_sample=False,
                    logits_processor=logits_processor,
                    use_cache=True,
                )

            responses = self.tokenizer.batch_decode(
                gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            contexts = self.tokenizer.batch_decode(
                input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            for idx, ctx, resp in zip(batch_idx, contexts, responses):
                out = resp.strip()
                if ctx and out.startswith(ctx):
                    out = out[len(ctx):].strip()
                local_outputs.append((idx, out))

            progress.update(len(batch_prompts))

        progress.close()

        # åˆ†å¸ƒå¼�è�šå�ˆ
        if is_distributed():
            gathered = [None] * self.world_size if self.rank == 0 else None
            self.logger.debug(
                f"Rank {self.rank} gathering outputs from all processes")
            dist.gather_object(local_outputs, gathered, dst=0)
            self.logger.debug(f"Rank {self.rank}: finished gathering.")

            if self.rank == 0:
                all_outputs = []
                for outs in gathered:
                    all_outputs.extend(outs)
                # ä¿�è¯�é¡ºåº�
                all_outputs.sort(key=lambda x: x[0])
                self.logger.debug("Rank 0: sorted all outputs.")
                return all_outputs
            else:
                return []
        else:
            # å�•æœºæ—¶ä¹ŸæŒ‰ç´¢å¼•æŽ’åº�
            return sorted(local_outputs, key=lambda x: x[0])

    def setup_device(self):
        if is_distributed():
            self.rank, self.local_rank, self.world_size = ddp_setup()
            if torch.npu.is_available():
                self.device = f"npu:{self.local_rank}"
            elif torch.cuda.is_available():
                self.device = f"cuda:{self.local_rank}"
            else:
                self.device = "cpu"
        else:
            self.rank, self.local_rank, self.world_size = 0, 0, 1
            self.device = (
                "npu:0"
                if torch.npu.is_available()
                else ("cuda:0" if torch.cuda.is_available() else "cpu")
            )

        if self.rank == 0:
            self.logger.info(
                f"World size={self.world_size}, device={self.device}")

    def load_model(self):
        if self.dtype == "fp16":
            torch_dtype = torch.float16
        elif self.dtype == "bf16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_path, padding_side="left", use_fast=True
        )

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.llm_path, torch_dtype=torch_dtype
        ).to(self.device)

        self.llm_model.eval()
        torch.set_grad_enabled(False)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        self.llm_model.config.pad_token_id = self.tokenizer.pad_token_id

        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()

        if self.rank == 0:
            self.logger.info(f"Model loaded with dtype: {torch_dtype}")


class OpenKEClient:
    """
    OpenKE 知识图谱嵌入模型调用客户端。
    通过加载 OpenKE 预训练的模型文件，提供三元组评分、头/尾实体预测等推理功能。
    """

    def __init__(self, path: str, model_name: str, rank: int = 0):
        """
        初始化 OpenKE 客户端。
        :param path: 预训练模型文件路径 (.pth)。
        :param model_name: 模型的名称 (例如 'RotatE', 'TransE')，必须与训练时使用的一致。
        :param rank: 分布式进程的 rank (默认为 0)。
        """
        self.logger = setup_logger(self.__class__.__name__)
        self.rank = rank
        self.device = self._get_device()
        rank_logger(self.logger, self.rank)(f"使用设备: {self.device}")

        # 加载 OpenKE 预训练模型
        self.kge_model = self._load_openke_model(path, model_name)
        self.kge_model.to(self.device)
        self.kge_model.eval() # 设置为评估模式

        # 在 OpenKE 模型对象中，实体和关系总数通常存储在 .ent_tot 和 .rel_tot
        self.ent_count = self.kge_model.ent_tot
        self.rel_count = self.kge_model.rel_tot

        rank_logger(self.logger, self.rank)(
            f"成功从 {path} 加载 KGE 模型 ({model_name})，包含 {self.ent_count} 个实体和 {self.rel_count} 个关系。"
        )

    def _get_device(self):
        """
        自动检测并返回可用设备 (NPU, CUDA, or CPU)。
        """
        if hasattr(torch, 'npu') and torch.npu.is_available():
            return torch.device(f'npu:{self.rank}')
        elif torch.cuda.is_available():
            return torch.device(f'cuda:{self.rank}')
        else:
            return torch.device('cpu')

    def _instantiate_and_load_model(self, model_name, state_dict):
        """
        根据模型名称实例化模型并加载参数。
        """
        # 动态导入 OpenKE 模型类
        try:
            from openke.module.model import RotatE
        except ImportError:
            self.logger.error("请确认您已经正确安装了 OpenKE 库 (pip install openke-torch)。")
            raise

        # 模型工厂
        model_classes = {'rotate': RotatE}

        model_class = model_classes.get(model_name.lower())
        if not model_class:
            raise ValueError(f"不支持的模型: {model_name}。支持的模型包括: {list(model_classes.keys())}")

        # 使用 OpenKE 的默认参数进行实例化。
        if model_name.lower() == 'rotate':
            # RotatE(ent_tot, rel_tot, dim, margin, epsilon)
            model = model_class(ent_tot=self.ent_tot, rel_tot=self.rel_tot, dim=self.dim, margin=self.margin, epsilon=2.0)
        else:
            raise ValueError(f"模型 {model_name} 的实例化参数未定义。")

        model.load_state_dict(state_dict)
        rank_logger(self.logger, self.rank)(f"已成功实例化 {model_name} 模型并加载参数。")
        return model

    def _load_openke_model(self, path: str, model_name: str):
        """
        从指定路径加载 OpenKE 预训练模型的状态字典，并返回一个完整的模型对象。
        """
        try:
            # 首先将 state_dict 加载到 CPU，以避免设备不匹配问题
            state_dict = torch.load(path, map_location='cpu')

            # 提取模型参数
            self.margin = state_dict.get('margin', 6.0)
            self.ent_tot = state_dict.get('ent_embeddings.weight').shape[0]
            self.rel_tot = state_dict.get('rel_embeddings.weight').shape[0]
            self.dim = state_dict.get('rel_embeddings.weight').shape[1]

            rank_logger(self.logger, self.rank)(f"已从 {path} 加载模型参数字典。")
            return self._instantiate_and_load_model(model_name, state_dict)
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            raise

    def _predict(self, h, r, t):
        """
        使用 OpenKE 模型内部的 predict 函数进行打分。
        h, r, t 应该是 LongTensor。
        """
        return -1 * self.kge_model.predict({
            "batch_h": h,
            "batch_r": r,
            "batch_t": t,
            "mode": "normal" # 某些模型需要 mode 参数
        })

    def get_score(self, head_id: int, rel_id: int, tail_id: int) -> float:
        """
        计算单个三元组 (h, r, t) 的得分。
        分数越高，表示该三元组越可能是事实。
        """
        h = torch.tensor([head_id], dtype=torch.long, device=self.device)
        r = torch.tensor([rel_id], dtype=torch.long, device=self.device)
        t = torch.tensor([tail_id], dtype=torch.long, device=self.device)
        score = self._predict(h, r, t)
        return score.item()

    def get_tail(self, head_id: int, rel_id: int, topk: int = 10):
        """
        给定头实体和关系，预测最可能的 Top-K 尾实体。
        """
        h = torch.tensor([head_id], dtype=torch.long, device=self.device).repeat(self.ent_count)
        r = torch.tensor([rel_id], dtype=torch.long, device=self.device).repeat(self.ent_count)
        all_tails = torch.arange(self.ent_count, dtype=torch.long, device=self.device)

        scores = self._predict(h, r, all_tails)
        _, topk_indices = torch.topk(scores, topk, largest=True)
        return topk_indices.tolist()

    def get_head(self, tail_id: int, rel_id: int, topk: int = 10):
        """
        给定尾实体和关系，预测最可能的 Top-K 头实体。
        """
        t = torch.tensor([tail_id], dtype=torch.long, device=self.device).repeat(self.ent_count)
        r = torch.tensor([rel_id], dtype=torch.long, device=self.device).repeat(self.ent_count)
        all_heads = torch.arange(self.ent_count, dtype=torch.long, device=self.device)

        scores = self._predict(all_heads, r, t)
        _, topk_indices = torch.topk(scores, topk, largest=True)
        return topk_indices.tolist()

    def batch_get_head(self, tails: List[int], rels: List[int], topk: int = 10):
        """
        批量预测头实体。
        """
        num_batches = len(tails)
        tails_tensor = torch.tensor(tails, dtype=torch.long, device=self.device).view(-1, 1)
        rels_tensor = torch.tensor(rels, dtype=torch.long, device=self.device).view(-1, 1)

        t = tails_tensor.expand(-1, self.ent_count)
        r = rels_tensor.expand(-1, self.ent_count)
        all_heads = torch.arange(self.ent_count, dtype=torch.long, device=self.device).expand(num_batches, -1)

        scores = self._predict(all_heads, r, t)
        _, topk_indices = torch.topk(scores, topk, dim=1, largest=True)
        return topk_indices.tolist()

    def batch_get_tail(self, heads: List[int], rels: List[int], topk: int = 10):
        """
        批量预测尾实体。
        """
        num_batches = len(heads)
        heads_tensor = torch.tensor(heads, dtype=torch.long, device=self.device).view(-1, 1)
        rels_tensor = torch.tensor(rels, dtype=torch.long, device=self.device).view(-1, 1)

        h = heads_tensor.expand(-1, self.ent_count)
        r = rels_tensor.expand(-1, self.ent_count)
        all_tails = torch.arange(self.ent_count, dtype=torch.long, device=self.device).expand(num_batches, -1)

        scores = self._predict(h, r, all_tails)
        _, topk_indices = torch.topk(scores, topk, dim=1, largest=True)
        return topk_indices.tolist()

    def get_top_p_heads(self, tail_id: int, rel_id: int, candidate_heads: List[int], p: float):
        """
        从给定的候选头实体列表中，选出得分最高的前 p% 的实体。
        """
        assert 0 < p <= 1, "参数 p 必须在 (0, 1] 范围内"
        assert candidate_heads, "候选头实体列表不能为空"

        num_candidates = len(candidate_heads)
        t = torch.tensor([tail_id], dtype=torch.long, device=self.device).repeat(num_candidates)
        r = torch.tensor([rel_id], dtype=torch.long, device=self.device).repeat(num_candidates)
        heads_tensor = torch.tensor(candidate_heads, dtype=torch.long, device=self.device)

        scores = self._predict(heads_tensor, r, t)

        k = max(1, math.ceil(num_candidates * p))
        _, topk_indices = torch.topk(scores, k, largest=True)

        result_node_ids = [candidate_heads[idx] for idx in topk_indices.tolist()]
        return result_node_ids

    def get_top_p_tails(self, head_id: int, rel_id: int, candidate_tails: List[int], p: float):
        """
        从给定的候选尾实体列表中，选出得分最高的前 p% 的实体。
        """
        assert 0 < p <= 1, "参数 p 必须在 (0, 1] 范围内"
        assert candidate_tails, "候选尾实体列表不能为空"

        num_candidates = len(candidate_tails)
        h = torch.tensor([head_id], dtype=torch.long, device=self.device).repeat(num_candidates)
        r = torch.tensor([rel_id], dtype=torch.long, device=self.device).repeat(num_candidates)
        tails_tensor = torch.tensor(candidate_tails, dtype=torch.long, device=self.device)

        scores = self._predict(h, r, tails_tensor)

        k = max(1, math.ceil(num_candidates * p))
        _, topk_indices = torch.topk(scores, k, largest=True)

        result_node_ids = [candidate_tails[idx] for idx in topk_indices.tolist()]
        return result_node_ids


if __name__=="__main__":
    pass
