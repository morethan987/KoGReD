import os
import httpx
from openai import OpenAI
import time
import math
from typing import List, Tuple, Optional
from tqdm.auto import tqdm
import torch.distributed as dist
from utils import is_distributed, ddp_setup, shard_indices
from setup_logger import setup_logger
from logits_processor import BinaryOutputProcessor
from transformers.generation.logits_process import LogitsProcessorList
from dotenv import load_dotenv
import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer


# åŠ è½½ .env æ–‡ä»¶
load_dotenv()


class RemoteLLM:
    """LLMæ¨¡åž‹è°ƒç”¨å®¢æˆ·ç«¯"""

    def __init__(self,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model_name: Optional[str] = None):
        """
        åˆ�å§‹åŒ–LLMå®¢æˆ·ç«¯

        Args:
            api_key: APIå¯†é’¥
            base_url: APIåŸºç¡€URL
            model_name: æ¨¡åž‹å��ç§°
        """
        self.logger = setup_logger(self.__class__.__name__)

        # ä»ŽçŽ¯å¢ƒå�˜é‡�æˆ–å�‚æ•°èŽ·å�–é…�ç½®
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL")
        self.model_name = model_name or os.getenv(
            "DEEPSEEK_MODEL_NAME", "deepseek-chat")

        # åˆ�å§‹åŒ–OpenAIå®¢æˆ·ç«¯
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=httpx.Client(trust_env=False)
            )
            self.logger.info(
                f"LLM client initialized with model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            raise

        # è°ƒç”¨ç»Ÿè®¡
        self.call_count = 0
        self.total_tokens = 0

    def get_output(self, prompt: str,
                   temperature: float = 0.0,
                   max_tokens: Optional[int] = None,
                   max_retries: int = 3,
                   retry_delay: float = 2.0) -> str:
        """
        èŽ·å�–LLMè¾“å‡º

        Args:
            prompt: è¾“å…¥æ��ç¤º
            temperature: ç”Ÿæˆ�æ¸©åº¦
            max_tokens: æœ€å¤§tokenæ•°
            max_retries: æœ€å¤§é‡�è¯•æ¬¡æ•°
            retry_delay: é‡�è¯•å»¶è¿Ÿï¼ˆç§’ï¼‰

        Returns:
            LLMç”Ÿæˆ�çš„æ–‡æœ¬
        """
        attempt = 0
        while attempt < max_retries:
            try:
                # å‡†å¤‡è¯·æ±‚
                messages = [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

                # å‡†å¤‡å�‚æ•°
                kwargs = {
                    "messages": messages,
                    "model": self.model_name,
                    "temperature": temperature
                }

                if max_tokens:
                    kwargs["max_tokens"] = max_tokens

                # è°ƒç”¨API
                self.logger.debug(
                    f"Calling LLM with prompt length: {len(prompt)}")
                response = self.client.chat.completions.create(**kwargs)

                # æ��å�–å“�åº”
                content = response.choices[0].message.content

                # æ›´æ–°ç»Ÿè®¡ä¿¡æ�¯
                self.call_count += 1
                if hasattr(response, 'usage') and response.usage:
                    self.total_tokens += response.usage.total_tokens

                self.logger.debug(
                    f"LLM call successful, response length: {len(content)}")
                return content

            except Exception as e:
                attempt += 1
                self.logger.warning(f"LLM call attempt {attempt} failed: {e}")

                if attempt < max_retries:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error("Max retries reached for LLM call")
                    raise

        return ""


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


class KGEModel:
    def __init__(self, path):
        """
        åˆ�å§‹åŒ– KGE æ¨¡åž‹ï¼Œä»ŽæŒ‡å®šè·¯å¾„åŠ è½½å®žä½“å’Œå…³ç³»åµŒå…¥åˆ°å¯¹åº”è®¾å¤‡ã€‚
        :param path: é¢„è®­ç»ƒæ¨¡åž‹æ–‡ä»¶è·¯å¾„ï¼ˆ.pthï¼‰
        """
        self.logger = setup_logger(self.__class__.__name__)
        # æ£€æµ‹å¹¶è®¾ç½®è®¾å¤‡
        self.device = self._get_device()
        print(f"ä½¿ç”¨è®¾å¤‡: {self.device}")

        # åŠ è½½æ¨¡åž‹åˆ°æŒ‡å®šè®¾å¤‡
        self.ent_embs, self.rel_embs = self.load_pretrain_kge(path)
        self.ent_embs = self.ent_embs.to(self.device)
        self.rel_embs = self.rel_embs.to(self.device)

        self.ent_count = self.ent_embs.size(0)
        self.rel_count = self.rel_embs.size(0)

    def _get_device(self):
        """
        è‡ªåŠ¨æ£€æµ‹å¹¶è¿”å›žå�¯ç”¨è®¾å¤‡ã€‚
        :return: torch.device
        """
        if hasattr(torch, 'npu') and torch.npu.is_available():
            return torch.device('npu')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def load_pretrain_kge(self, path):
        """
        åŠ è½½é¢„è®­ç»ƒçš„ KGE æ¨¡åž‹ä¸­çš„å®žä½“å’Œå…³ç³»åµŒå…¥ã€‚
        :param path: æ¨¡åž‹è·¯å¾„
        :return: ent_embs, rel_embs (å�‡ä¸º torch.Tensor)
        """
        # æ ¹æ�®è®¾å¤‡è®¾ç½® map_location
        if self.device.type == 'cuda':
            map_location = 'cuda'
        elif self.device.type == 'npu':
            map_location = 'npu'
        else:
            map_location = 'cpu'

        self.logger.info(f"åœ¨ {self.device} è®¾å¤‡ä¸ŠåŠ è½½ KGE æ¨¡åž‹")
        kge_model = torch.load(path, map_location=map_location)

        ent_embs = kge_model["ent_embeddings.weight"].clone().detach()
        rel_embs = kge_model["rel_embeddings.weight"].clone().detach()

        ent_embs.requires_grad_(False)
        rel_embs.requires_grad_(False)

        ent_dim = ent_embs.shape[1]
        rel_dim = rel_embs.shape[1]

        self.logger.debug(f"å®žä½“ç»´åº¦: {ent_dim}, å…³ç³»ç»´åº¦: {rel_dim}")

        # å¦‚æžœç»´åº¦ä¸�å�Œï¼Œåˆ™å¤�åˆ¶ rel_embs ä½¿å…¶ä¸Ž ent_embs ç»´åº¦ä¸€è‡´
        if ent_dim != rel_dim:
            rel_embs = torch.cat((rel_embs, rel_embs), dim=-1)

        return ent_embs, rel_embs

    def score_triplet(self, head_emb, rel_emb, tail_emb):
        """
        è®¡ç®—ä¸‰å…ƒç»„ (h, r, t) çš„å¾—åˆ†ã€‚ä½¿ç”¨ RotatE é£Žæ ¼çš„è¯„åˆ†æ–¹å¼�ã€‚
        :param head_emb: [*, dim]
        :param rel_emb: [*, dim]
        :param tail_emb: [*, dim]
        :return: [*, 1] åˆ†æ•°
        """
        # RotatE é£Žæ ¼è¯„åˆ†ï¼š||h * r - t||
        score = head_emb * rel_emb - tail_emb
        return -torch.norm(score, p=2, dim=-1)  # è¶Šå°�è¶Šå¥½ï¼Œæ‰€ä»¥åŠ è´Ÿå�·

    def get_tail(self, head_id: int, rel_id: int, topk: int = 10):
        """
        å›ºå®šå¤´å®žä½“å’Œå…³ç³»ï¼Œè¿”å›ž Top-K æœ€å�¯èƒ½çš„å°¾å®žä½“ IDã€‚
        :param head_id: å¤´å®žä½“ ID
        :param rel_id: å…³ç³» ID
        :param topk: è¿”å›ž top-k ä¸ªå°¾å®žä½“
        :return: topk å°¾å®žä½“ ID åˆ—è¡¨
        """
        h = self.ent_embs[head_id].unsqueeze(0)  # [1, dim]
        r = self.rel_embs[rel_id].unsqueeze(0)   # [1, dim]
        all_tails = self.ent_embs                # [num_ent, dim]

        scores = self.score_triplet(h, r, all_tails)  # [num_ent]
        topk_scores, topk_indices = torch.topk(scores, topk, largest=True)
        return topk_indices.tolist()

    def get_head(self, tail_id: int, rel_id: int, topk: int = 10):
        """
        å›ºå®šå°¾å®žä½“å’Œå…³ç³»ï¼Œè¿”å›ž Top-K æœ€å�¯èƒ½çš„å¤´å®žä½“ IDã€‚
        :param tail_id: å°¾å®žä½“ ID
        :param rel_id: å…³ç³» ID
        :param topk: è¿”å›ž top-k ä¸ªå¤´å®žä½“
        :return: topk å¤´å®žä½“ ID åˆ—è¡¨
        """
        t = self.ent_embs[tail_id].unsqueeze(0)  # [1, dim]
        r = self.rel_embs[rel_id].unsqueeze(0)   # [1, dim]
        all_heads = self.ent_embs                # [num_ent, dim]

        # RotatE å��å�‘æŽ¨ç�†ï¼šh * r â‰ˆ t => h â‰ˆ t / r
        # ç®€åŒ–å¤„ç�†ä½¿ç”¨ -r è¿›è¡Œè¿‘ä¼¼è®¡ç®—
        scores = self.score_triplet(all_heads, -r, t)  # [num_ent]
        topk_scores, topk_indices = torch.topk(scores, topk, largest=True)
        return topk_indices.tolist()

    def batch_get_tail(self, heads: list, rels: list, topk: int = 10):
        """
        æ‰¹é‡�èŽ·å�–å°¾å®žä½“ Top-Kã€‚
        :param heads: å¤´å®žä½“ ID åˆ—è¡¨
        :param rels: å…³ç³» ID åˆ—è¡¨
        :param topk: æ¯�ä¸ªæ ·æœ¬è¿”å›ž top-k ä¸ªå°¾å®žä½“
        :return: List[List[int]]ï¼Œæ¯�ä¸ªå­�åˆ—è¡¨åŒ…å�« topk ä¸ªå°¾å®žä½“ id
        """
        heads_tensor = torch.tensor(
            heads, dtype=torch.long, device=self.device)  # [B]
        rels_tensor = torch.tensor(
            rels, dtype=torch.long, device=self.device)     # [B]

        h = self.ent_embs[heads_tensor]  # [B, dim]
        r = self.rel_embs[rels_tensor]   # [B, dim]
        all_tails = self.ent_embs        # [num_ent, dim]

        # æ‰©å±• h å’Œ r åˆ° [B, num_ent, dim]
        h = h.unsqueeze(1)  # [B, 1, dim]
        r = r.unsqueeze(1)  # [B, 1, dim]
        tails_exp = all_tails.unsqueeze(0)  # [1, num_ent, dim]

        scores = self.score_triplet(h, r, tails_exp)  # [B, num_ent]
        _, topk_indices = torch.topk(
            scores, topk, dim=1, largest=True)  # [B, topk]

        return topk_indices.tolist()

    def batch_get_head(self, tails: list, rels: list, topk: int = 10):
        """
        æ‰¹é‡�èŽ·å�–å¤´å®žä½“ Top-Kã€‚
        :param tails: å°¾å®žä½“ ID åˆ—è¡¨
        :param rels: å…³ç³» ID åˆ—è¡¨
        :param topk: æ¯�ä¸ªæ ·æœ¬è¿”å›ž top-k ä¸ªå¤´å®žä½“
        :return: List[List[int]]
        """
        tails_tensor = torch.tensor(
            tails, dtype=torch.long, device=self.device)
        rels_tensor = torch.tensor(
            rels, dtype=torch.long, device=self.device)

        t = self.ent_embs[tails_tensor]  # [B, dim]
        r = self.rel_embs[rels_tensor]   # [B, dim]
        all_heads = self.ent_embs        # [num_ent, dim]

        t = t.unsqueeze(1)  # [B, 1, dim]
        r = r.unsqueeze(1)  # [B, 1, dim]
        heads_exp = all_heads.unsqueeze(0)  # [1, num_ent, dim]

        scores = self.score_triplet(heads_exp, -r, t)  # [B, num_ent]
        _, topk_indices = torch.topk(
            scores, topk, dim=1, largest=True)  # [B, topk]

        return topk_indices.tolist()

    def get_top_p_tails(self, head_id: int, rel_id: int, candidate_tails: list, p: float):
        """
        å›ºå®šå¤´å®žä½“å’Œå…³ç³»ï¼Œä»Žå€™é€‰å°¾å®žä½“ä¸­é€‰å‡ºå¾—åˆ†æœ€é«˜çš„å‰� p æ¯”ä¾‹èŠ‚ç‚¹ã€‚

        :param head_id: å¤´å®žä½“ ID
        :param rel_id: å…³ç³» ID
        :param candidate_tails: å€™é€‰å°¾å®žä½“ ID åˆ—è¡¨
        :param p: æ¯”ä¾‹ï¼ŒèŒƒå›´ (0, 1]
        :return: å‰� p æ¯”ä¾‹çš„å°¾å®žä½“ ID åˆ—è¡¨ï¼ˆå·²æŽ’åº�ï¼Œå¾—åˆ†ä»Žé«˜åˆ°ä½Žï¼‰
        """
        assert 0 < p <= 1, "å�‚æ•° p å¿…é¡»åœ¨ (0, 1] èŒƒå›´å†…"
        assert candidate_tails, "å€™é€‰å°¾å®žä½“åˆ—è¡¨ä¸�èƒ½ä¸ºç©º"

        h = self.ent_embs[head_id].unsqueeze(0)  # [1, dim]
        r = self.rel_embs[rel_id].unsqueeze(0)   # [1, dim]
        candidate_tensor = torch.tensor(
            candidate_tails, dtype=torch.long, device=self.device)
        tails = self.ent_embs[candidate_tensor]  # [num_candidates, dim]

        scores = self.score_triplet(h, r, tails)  # [num_candidates]

        # è®¡ç®—è¦�è¿”å›žçš„ Top-K æ•°é‡�ï¼ˆå�‘ä¸Šå�–æ•´ï¼Œè‡³å°‘ä¸º1ï¼‰
        k = max(1, math.ceil(len(candidate_tails) * p))

        # èŽ·å�– Top-K çš„ç´¢å¼•ï¼ˆå¾—åˆ†ä»Žé«˜åˆ°ä½Žï¼‰
        topk_scores, topk_indices = torch.topk(scores, k, largest=True)

        # æ˜ å°„å›žåŽŸå§‹ candidate ID
        result_node_ids = [candidate_tails[idx]
                           for idx in topk_indices.tolist()]

        return result_node_ids

    def get_top_p_heads(self, tail_id: int, rel_id: int, candidate_heads: list, p: float):
        """
        å›ºå®šå°¾å®žä½“å’Œå…³ç³»ï¼Œä»Žå€™é€‰å¤´å®žä½“ä¸­é€‰å‡ºå¾—åˆ†æœ€é«˜çš„å‰� p æ¯”ä¾‹èŠ‚ç‚¹ã€‚

        :param tail_id: å°¾å®žä½“ ID
        :param rel_id: å…³ç³» ID
        :param candidate_heads: å€™é€‰å¤´å®žä½“ ID åˆ—è¡¨
        :param p: æ¯”ä¾‹ï¼ŒèŒƒå›´ (0, 1]
        :return: å‰� p æ¯”ä¾‹çš„å¤´å®žä½“ ID åˆ—è¡¨ï¼ˆå·²æŽ’åº�ï¼Œå¾—åˆ†ä»Žé«˜åˆ°ä½Žï¼‰
        """
        assert 0 < p <= 1, "å�‚æ•° p å¿…é¡»åœ¨ (0, 1] èŒƒå›´å†…"
        assert candidate_heads, "å€™é€‰å¤´å®žä½“åˆ—è¡¨ä¸�èƒ½ä¸ºç©º"

        t = self.ent_embs[tail_id].unsqueeze(0)  # [1, dim]
        r = self.rel_embs[rel_id].unsqueeze(0)   # [1, dim]
        candidate_tensor = torch.tensor(
            candidate_heads, dtype=torch.long, device=self.device)
        heads = self.ent_embs[candidate_tensor]  # [num_candidates, dim]

        # RotatE å��å�‘æŽ¨ç�†ï¼šh * r â‰ˆ t => h â‰ˆ t / rï¼Œç®€åŒ–ä½¿ç”¨ -r
        scores = self.score_triplet(heads, -r, t)  # [num_candidates]

        # è®¡ç®—è¦�è¿”å›žçš„ Top-K æ•°é‡�ï¼ˆå�‘ä¸Šå�–æ•´ï¼Œè‡³å°‘ä¸º1ï¼‰
        k = max(1, math.ceil(len(candidate_heads) * p))

        # èŽ·å�– Top-K çš„ç´¢å¼•ï¼ˆå¾—åˆ†ä»Žé«˜åˆ°ä½Žï¼‰
        topk_scores, topk_indices = torch.topk(scores, k, largest=True)

        # æ˜ å°„å›žåŽŸå§‹ candidate ID
        result_node_ids = [candidate_heads[idx]
                           for idx in topk_indices.tolist()]

        return result_node_ids
