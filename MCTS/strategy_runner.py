import os
import sys
import json
import torch
import argparse
import threading
from setup_logger import setup_logger, rank_logger
from utils import init_distributed, cleanup_distributed, shard_indices
import torch.distributed as dist
from tqdm.auto import tqdm
from transformers.utils import logging
logging.set_verbosity_error()


class StrategyRunner:
    """
    统一搜索策略运行入口。
    支持 random / greedy / beam / mcts 四种策略，
    共享相同的数据加载、判别器和预算控制。
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.data_folder = args.data_folder
        self.logger = setup_logger(self.__class__.__name__)
        self.checkpoint_thread = None
        self.strategy_name = args.strategy

        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        self.is_initialed = init_distributed(
            self.rank, self.local_rank, self.world_size
        )

        self.processed_data = torch.load(args.processed_data)
        self.get_unprocessed(self.world_size)
        self.checkpoint_file = os.path.join(
            args.output_folder,
            f"checkpoints/checkpoint_rank_{self.rank}.json"
        )

        self.data_loader, self.kge_model, self.triplet_discriminator = (
            self._init_components()
        )

        self.all_entities = set(self.data_loader.entity2name.keys())

        if self.strategy_name == "mcts":
            self.mcts_enhancer = self._init_mcts_enhancer()
        else:
            from search_strategies import STRATEGY_REGISTRY
            strategy_cls = STRATEGY_REGISTRY[self.strategy_name]
            self.strategy = strategy_cls(
                rank=self.rank,
                **({"beam_width": args.beam_width} if self.strategy_name == "beam" else {})
            )

        self.all_discovered_triplets = set()
        self.processed_entities = set()

        self.load_checkpoint()

    # ------------------------------------------------------------------
    # 组件初始化
    # ------------------------------------------------------------------

    def _init_components(self):
        from kg_data_loader import KGDataLoader
        from model_calls import OpenKEClient

        data_loader = KGDataLoader(
            entity2name_path=f"{self.data_folder}/entity2name.txt",
            entity2embedding_path=self.args.entity2embedding_path,
            relation2id_path=f"{self.data_folder}/relation2id.txt",
            entity2id_path=f"{self.data_folder}/entity2id.txt",
            entity2description_path=f"{self.data_folder}/entity2des.txt",
            kg_path=f"{self.data_folder}/train.txt",
        )

        kge_model = OpenKEClient(
            path=self.args.kge_path,
            model_name="RotatE",
            rank=self.rank,
        )

        triplet_discriminator = self._create_discriminator(data_loader)

        return data_loader, kge_model, triplet_discriminator

    def _create_discriminator(self, data_loader):
        dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        torch_dtype = dtype_map.get(self.args.dtype, torch.float32)

        disc_type = self.args.discriminator_type

        if disc_type == "llm":
            from LLM_Discriminator.discriminator import TriplesDiscriminator
            discriminator = TriplesDiscriminator(
                llm_path=self.args.llm_path,
                lora_path=self.args.lora_path,
                embedding_path=self.args.embedding_path,
                device=self.device,
                dtype=torch_dtype,
                batch_size=16,
            )
            calibration_data = self._prepare_calibration_data(data_loader)
            if calibration_data:
                import random as _random
                max_calibration = 2000
                if len(calibration_data) > max_calibration:
                    _random.shuffle(calibration_data)
                    calibration_data = calibration_data[:max_calibration]
                discriminator.calibrate(calibration_data)
            return discriminator

        elif disc_type == "kgbert":
            from kgbert_discriminator import KGBERTDiscriminator
            discriminator = KGBERTDiscriminator(
                model_dir=self.args.kgbert_model_dir,
                data_dir=self.args.kgbert_data_dir,
                batch_size=16,
                device=self.device,
            )
            discriminator.set_id_mappings(
                id2entity=data_loader.id2entity,
                id2relation=data_loader.id2relation,
            )
            return discriminator

        elif disc_type == "kge":
            from kge_discriminator import KGEDiscriminator
            return KGEDiscriminator(
                model_path=self.args.kge_discriminator_path,
                model_name="RotatE",
                device=self.device,
                batch_size=16,
            )

        elif disc_type == "random":
            from random_discriminator import RandomDiscriminator
            return RandomDiscriminator(positive_rate=0.5)

        else:
            raise ValueError(f"Unknown discriminator type: {disc_type}")

    def _prepare_calibration_data(self, data_loader, num_neg_per_positive: int = 10):
        import random

        valid_path = f"{self.data_folder}/valid.txt"
        if not os.path.isfile(valid_path):
            return []

        entity2id = data_loader.entity2id
        relation2id = data_loader.relation2id
        entity2name = data_loader.entity2name
        all_entity_ids = list(entity2id.values())

        calibration_samples = []

        with open(valid_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    continue
                h_str, r_str, t_str = parts
                h_id = entity2id.get(h_str)
                r_id = relation2id.get(r_str)
                t_id = entity2id.get(t_str)
                if h_id is None or r_id is None or t_id is None:
                    continue

                input_text = "The input triple: \n( {head}, {rel}, {tail} )\n".format(
                    head=entity2name.get(h_str, h_str).replace('_', ' '),
                    rel=r_str.replace('/', ' '),
                    tail=entity2name.get(t_str, t_str).replace('_', ' '),
                )
                calibration_samples.append({
                    "input": input_text,
                    "embedding_ids": [h_id, r_id, t_id],
                    "label": 1,
                })

                for _ in range(num_neg_per_positive):
                    neg_t_id = random.choice(all_entity_ids)
                    if neg_t_id == t_id:
                        continue
                    neg_t_str = data_loader.id2entity.get(neg_t_id, "")
                    if not neg_t_str:
                        continue
                    neg_input_text = "The input triple: \n( {head}, {rel}, {tail} )\n".format(
                        head=entity2name.get(h_str, h_str).replace('_', ' '),
                        rel=r_str.replace('/', ' '),
                        tail=entity2name.get(neg_t_str, neg_t_str).replace('_', ' '),
                    )
                    calibration_samples.append({
                        "input": neg_input_text,
                        "embedding_ids": [h_id, r_id, neg_t_id],
                        "label": 0,
                    })

        return calibration_samples

    def _init_mcts_enhancer(self):
        from kg_enhancer import KGEnhancer
        return KGEnhancer(
            rank=self.rank,
            entity2name_path=f"{self.data_folder}/entity2name.txt",
            relation2id_path=f"{self.data_folder}/relation2id.txt",
            entity2id_path=f"{self.data_folder}/entity2id.txt",
            output_folder=self.args.output_folder,
            entity2embedding_path=self.args.entity2embedding_path,
            entity2description_path=f"{self.data_folder}/entity2des.txt",
            kg_path=f"{self.data_folder}/train.txt",
            budget_per_entity=self.args.budget_per_entity,
            mcts_iterations=self.args.mcts_iterations,
            leaf_threshold=self.args.leaf_threshold,
            exploration_weight=self.args.exploration_weight,
            llm_path=self.args.llm_path,
            lora_path=self.args.lora_path,
            embedding_path=self.args.embedding_path,
            kge_path=self.args.kge_path,
            dtype=self.args.dtype,
            device=self.device,
            discriminator_type=self.args.discriminator_type,
            kgbert_model_dir=self.args.kgbert_model_dir,
            kgbert_data_dir=self.args.kgbert_data_dir,
            kge_discriminator_path=self.args.kge_discriminator_path,
            valid_path=f"{self.data_folder}/valid.txt",
            target_depth=self.args.target_depth,
        )

    # ------------------------------------------------------------------
    # 搜索执行（非 MCTS）
    # ------------------------------------------------------------------

    def _run_strategy_for_entity(
        self, entity: str, position: str, relation: str
    ) -> set:
        from node import SearchRootNode, Context

        candidate_entities = self.all_entities - {entity}

        context = Context(
            rank=self.rank,
            sparse_entity=entity,
            position=position,
            relation=relation,
            unfiltered_entities=candidate_entities,
            output_folder=self.args.output_folder,
            data_loader=self.data_loader,
            triplet_discriminator=self.triplet_discriminator,
            kge_model=self.kge_model,
            leaf_threshold=self.args.leaf_threshold,
            parent=None,
        )

        root_node = SearchRootNode(
            context=context, target_depth=self.args.target_depth
        )

        discovered, _ = self.strategy.search(
            root_node, self.args.budget_per_entity
        )
        return discovered

    # ------------------------------------------------------------------
    # 检查点
    # ------------------------------------------------------------------

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_file):
            return
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.processed_entities = set(data.get("processed_entities", []))
                saved_triplets = data.get("discovered_triplets", [])
                self.local_discovered_triplets.update(saved_triplets)
            rank_logger(self.logger, self.rank)(
                f"Loaded checkpoint: {len(self.processed_entities)} entities, "
                f"{len(saved_triplets)} triplets."
            )
        except Exception as e:
            rank_logger(self.logger, self.rank)(
                f"Failed to load checkpoint: {e}, starting from scratch."
            )

    def _perform_save(self, data, filepath):
        try:
            temp_filepath = filepath + ".tmp"
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.rename(temp_filepath, filepath)
            rank_logger(self.logger, self.rank)(
                "Checkpoint successfully saved in the background."
            )
        except Exception as e:
            rank_logger(self.logger, self.rank)(f"Background save failed: {e}")

    def save_checkpoint(self):
        if self.checkpoint_thread and self.checkpoint_thread.is_alive():
            rank_logger(self.logger, self.rank)(
                "Previous checkpoint save still in progress. Skipping."
            )
            return

        data = {
            "processed_entities": list(self.processed_entities),
            "discovered_triplets": list(self.local_discovered_triplets),
            "strategy": self.strategy_name,
            "entity_count": len(self.processed_entities),
            "triplet_count": len(self.local_discovered_triplets),
        }

        self.checkpoint_thread = threading.Thread(
            target=self._perform_save,
            args=(data, self.checkpoint_file),
        )
        self.checkpoint_thread.start()

    def get_unprocessed(self, device_num: int):
        processed = set()
        for i in range(device_num):
            file = os.path.join(
                self.args.output_folder,
                f"checkpoints/checkpoint_rank_{i}.json"
            )
            if not os.path.exists(file):
                continue
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                processed.update(data.get("processed_entities", []))
        self.processed_data = {
            k: v for k, v in self.processed_data.items() if k not in processed
        }

    # ------------------------------------------------------------------
    # 主运行逻辑
    # ------------------------------------------------------------------

    def run(self):
        items = list(self.processed_data.items())
        indices = shard_indices(len(items), self.rank, self.world_size)
        local_data = [items[i] for i in indices]
        self.logger.info(
            f"Rank {self.rank} processing {len(local_data)} entities "
            f"with strategy={self.strategy_name}."
        )

        self.local_discovered_triplets = set()

        progress = tqdm(
            total=len(local_data),
            desc=f"[{self.strategy_name}] rank {self.rank}",
            disable=(self.rank != 0),
        )

        processed_count_since_last_save = 0

        for entity_idx, (entity, position_relations) in enumerate(local_data):
            if entity in self.processed_entities:
                progress.update(1)
                continue

            rank_logger(self.logger, self.rank)(f"\n{'='*50}")
            rank_logger(self.logger, self.rank)(
                f"[{self.strategy_name}] Processing entity {entity_idx + 1}/{len(items)}: {entity}"
            )

            for pos_rel_idx, (position, relation) in enumerate(position_relations):
                rank_logger(self.logger, self.rank)(
                    f"  Pair {pos_rel_idx + 1}/{len(position_relations)}: "
                    f"position={position}, relation={relation}"
                )

                if self.strategy_name == "mcts":
                    discovered = self.mcts_enhancer.enhance_entity_relation(
                        entity, position, relation
                    )
                else:
                    discovered = self._run_strategy_for_entity(
                        entity, position, relation
                    )

                self.local_discovered_triplets.update(discovered)

                rank_logger(self.logger, self.rank)(
                    f"  Discovered {len(discovered)} valid triplets"
                )

            self.processed_entities.add(entity)
            processed_count_since_last_save += 1

            if processed_count_since_last_save >= self.args.checkpoint_interval:
                self.save_checkpoint()
                processed_count_since_last_save = 0

            progress.update(1)

        progress.close()
        self.save_checkpoint()

        if self.checkpoint_thread and self.checkpoint_thread.is_alive():
            self.checkpoint_thread.join()

        # 收集分布式结果
        if self.is_initialed:
            self.logger.info(f"Rank {self.rank} gathering results...")
            dist.barrier()
            gathered = [None] * self.world_size if self.rank == 0 else None
            dist.gather_object(
                list(self.local_discovered_triplets), gathered, dst=0
            )
            if self.rank == 0:
                for triplet_list in gathered:
                    self.all_discovered_triplets.update(triplet_list)
            else:
                return
        else:
            self.all_discovered_triplets = self.local_discovered_triplets

        # 保存结果
        output_path = os.path.join(
            self.args.output_folder, "discovered_triplets.txt"
        )
        os.makedirs(self.args.output_folder, exist_ok=True)
        rank_logger(self.logger, self.rank)(
            f"\nSaving {len(self.all_discovered_triplets)} discovered triplets to {output_path}"
        )
        with open(output_path, 'w', encoding='utf-8') as f:
            for head, rel, tail in set(self.all_discovered_triplets):
                f.write(f"{head}\t{rel}\t{tail}\n")

        self.logger.info(
            f"Rank {self.rank}: strategy={self.strategy_name} completed!"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search Strategy Ablation Runner"
    )

    parser.add_argument(
        "--strategy", type=str, required=True,
        choices=["random", "greedy", "beam", "mcts"],
        help="Search strategy to evaluate"
    )
    parser.add_argument(
        "--data_folder", type=str, required=True,
        help="Path to the dataset folder"
    )
    parser.add_argument(
        "--processed_data", type=str, required=True,
        help="Path to preprocessed sparse-entity data (.pth)"
    )
    parser.add_argument(
        "--output_folder", type=str, default="MCTS/output",
        help="Output folder"
    )
    parser.add_argument(
        "--llm_path", type=str, default=None,
        help="Path to the LLM model (required when --discriminator_type llm)"
    )
    parser.add_argument(
        "--lora_path", type=str, default=None,
        help="Path to the LoRA weights"
    )
    parser.add_argument(
        "--embedding_path", type=str, default=None,
        help="Path to the kg embeddings (for LLM discriminator)"
    )
    parser.add_argument(
        "--entity2embedding_path", type=str, required=True,
        help="entity2embedding file path"
    )
    parser.add_argument(
        "--kge_path", type=str, required=True,
        help="Path to the KGE model (for filtering)"
    )
    parser.add_argument(
        "--discriminator_folder", type=str, required=True,
        help="Discriminator module folder (added to sys.path)"
    )
    parser.add_argument(
        "--root_dir", type=str, default=".",
        help="Root directory for imports"
    )
    parser.add_argument(
        "--dtype", type=str, default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Data type for the model"
    )
    parser.add_argument(
        "--exploration_weight", type=float, default=1.0,
        help="Exploration weight for MCTS"
    )
    parser.add_argument(
        "--leaf_threshold", type=int, default=32,
        help="Threshold for leaf node"
    )
    parser.add_argument(
        "--mcts_iterations", type=int, default=10,
        help="Number of MCTS iterations"
    )
    parser.add_argument(
        "--budget_per_entity", type=int, default=200,
        help="Budget per sparse entity"
    )
    parser.add_argument(
        "--beam_width", type=int, default=3,
        help="Beam width for beam search"
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, default=10,
        help="Save checkpoint every N entities"
    )
    parser.add_argument(
        "--discriminator_type", type=str, default="kgbert",
        choices=["llm", "kgbert", "kge", "random"],
        help="Discriminator type"
    )
    parser.add_argument(
        "--kgbert_model_dir", type=str, default=None,
        help="Path to trained KG-BERT model dir"
    )
    parser.add_argument(
        "--kgbert_data_dir", type=str, default=None,
        help="Path to KG-BERT data dir"
    )
    parser.add_argument(
        "--kge_discriminator_path", type=str, default=None,
        help="Path to KGE model checkpoint for discriminator"
    )
    parser.add_argument(
        "--target_depth", type=int, default=4,
        help="Target search depth for adaptive filtering"
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    # 条件校验
    if args.discriminator_type == "llm":
        missing = [f"--{k}" for k in ["llm_path", "lora_path", "embedding_path"]
                   if getattr(args, k) is None]
        if missing:
            parser.error(f"discriminator_type=llm requires: {', '.join(missing)}")
    elif args.discriminator_type == "kgbert":
        missing = [f"--{k}" for k in ["kgbert_model_dir", "kgbert_data_dir"]
                   if getattr(args, k) is None]
        if missing:
            parser.error(f"discriminator_type=kgbert requires: {', '.join(missing)}")
    elif args.discriminator_type == "kge":
        if args.kge_discriminator_path is None:
            parser.error("discriminator_type=kge requires: --kge_discriminator_path")

    sys.path.append(args.discriminator_folder)
    sys.path.append(args.root_dir)

    runner = StrategyRunner(args)
    runner.run()

    cleanup_distributed()
