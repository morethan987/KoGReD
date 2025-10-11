import os
import sys
import json
import torch
import argparse
from setup_logger import setup_logger, rank_logger
from utils import init_distributed, cleanup_distributed, shard_indices, get_device
import torch.distributed as dist
from tqdm.auto import tqdm
from transformers.utils import logging
logging.set_verbosity_error()


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.data_folder = args.data_folder
        self.logger = setup_logger(self.__class__.__name__)

        # 获取分布式信息
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.device, self.device_type = get_device(self.local_rank)
        if self.device_type == "npu":
            torch.npu.set_device(self.device)
        elif self.device_type == "cuda":
            torch.cuda.set_device(self.device)

        # 初始化分布式环境
        self.is_initialed = init_distributed(
            self.rank, self.local_rank, self.world_size
        )

        self.processed_data = torch.load(args.processed_data)
        self.get_unprocessed(self.world_size)
        self.checkpoint_file = os.path.join(args.output_folder, f"checkpoints/checkpoint_rank_{self.rank}.json")

        self.enhancer = KGEnhancer(
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
            device=self.device)

        self.all_discovered_triplets = []
        self.processed_entities = set()
        self.local_discovered_triplets = []

        # 加载检查点
        self.load_checkpoint()

    def load_checkpoint(self):
        """加载检查点"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.processed_entities = set(data.get("processed_entities", []))
                    # 加载已发现的三元组
                    saved_triplets = data.get("discovered_triplets", [])
                    self.local_discovered_triplets.extend(saved_triplets)
                    # 新增：加载并恢复策略模型的状态
                    if "rollout_policy_state" in data:
                        policy_state = data["rollout_policy_state"]
                        self.enhancer.rollout_policy.load_state(policy_state)
                    else:
                        rank_logger(self.logger, self.rank)("No rollout policy state found in checkpoint. Initializing a new policy.")

                rank_logger(self.logger, self.rank)(f"Loaded checkpoint: {len(self.processed_entities)} entities processed, {len(saved_triplets)} triplets discovered.")
            except Exception as e:
                rank_logger(self.logger, self.rank)(f"Failed to load checkpoint: {e}, starting from scratch.")

    def save_checkpoint(self):
        """保存检查点"""
        policy_state = self.enhancer.rollout_policy.get_state()
        data = {
            "processed_entities": list(self.processed_entities),
            "discovered_triplets": self.local_discovered_triplets,
            "rollout_policy_state": policy_state,
            "entity_count": len(self.processed_entities),
            "triplet_count": len(self.local_discovered_triplets)
        }
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            rank_logger(self.logger, self.rank)(f"Checkpoint saved: {len(self.processed_entities)} entities processed, {len(self.local_discovered_triplets)} triplets discovered.")
        except Exception as e:
            rank_logger(self.logger, self.rank)(f"Failed to save checkpoint: {e}")

    def get_unprocessed(self, device_num: int):
        """过滤掉已经处理过的实体"""
        processed = set()
        for i in range(device_num):
            file = os.path.join(args.output_folder, f"checkpoints/checkpoint_rank_{i}.json")
            if not os.path.exists(file):
                continue
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                processed.update(data.get("processed_entities", []))
        self.processed_data = {k: v for k, v in self.processed_data.items() if k not in processed}

    def run(self):
        items = list(self.processed_data.items())
        indices = shard_indices(len(items), self.rank, self.world_size)
        local_data = [items[i] for i in indices]
        self.logger.info(f"Rank {self.rank} processing {len(local_data)} entities.")

        progress = tqdm(
            total=len(local_data),
            desc=f"Processing (rank {self.rank})",
            disable=(self.rank != 0),
        )

        checkpoint_counter = 0
        processed_count_since_last_save = 0

        for entity_idx, (entity, position_relations) in enumerate(local_data):
            if entity in self.processed_entities:
                rank_logger(self.logger, self.rank)(f"Skipping already processed entity: {entity}")
                progress.update(1)
                continue

            rank_logger(self.logger, self.rank)(f"\n{'='*50}")
            rank_logger(self.logger, self.rank)(
                f"Processing sparse entity {entity_idx + 1}/{len(items)}: {entity}"
            )
            rank_logger(self.logger, self.rank)(
                f"Position-relation pairs: {len(position_relations)}")

            entity_triplets = []
            for pos_rel_idx, (position, relation) in enumerate(position_relations):
                rank_logger(self.logger, self.rank)(
                    f"\nProcessing pair {pos_rel_idx + 1}/{len(position_relations)}: position={position}, relation={relation}")

                discovered = self.enhancer.enhance_entity_relation(
                    entity, position, relation
                )

                entity_triplets.extend(discovered)
                self.local_discovered_triplets.extend(discovered)
                rank_logger(self.logger, self.rank)(
                    f"Discovered {len(discovered)} valid triplets for {entity}-{position}-{relation}")

            # 标记为已处理
            self.processed_entities.add(entity)
            processed_count_since_last_save += 1

            # 每处理一定数量的实体保存一次检查点
            if processed_count_since_last_save >= self.args.checkpoint_interval:
                self.save_checkpoint()
                processed_count_since_last_save = 0

            progress.update(1)

        progress.close()

        # 保存最终检查点
        self.save_checkpoint()

        # 收集所有进程的结果
        if self.is_initialed:
            self.logger.info(f"Rank {self.rank} gathering results...")
            dist.barrier()
            gathered = [None] * self.world_size if self.rank == 0 else None
            dist.gather_object(self.local_discovered_triplets, gathered, dst=0)
            if self.rank == 0:
                for triplet_list in gathered:
                    self.all_discovered_triplets.extend(tuple(triplet_list))
            else:
                return
        else:
            self.all_discovered_triplets = self.local_discovered_triplets

        # 保存所有发现的三元组
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

        self.logger.info(f"Rank {self.rank}: Knowledge graph enhancement completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCTS Runner")
    parser.add_argument(
        "--data_folder", type=str, required=True, help="Path to the dataset folder"
    )
    parser.add_argument(
        "--processed_data",
        type=str,
        required=True,
        help="Path to save/load preprocessed relations",
    )
    parser.add_argument(
        "--output_folder", type=str, default="MCTS/output", help="Output folder"
    )
    parser.add_argument(
        "--llm_path", type=str, required=True, help="Path to the LLM model"
    )
    parser.add_argument(
        "--lora_path", type=str, required=True, help="Path to the LoRA weights"
    )
    parser.add_argument(
        "--embedding_path", type=str, required=True, help="Path to the kg embeddings"
    )
    parser.add_argument(
        "--entity2embedding_path", type=str, required=True, help="entity2embedding file path",
    )
    parser.add_argument(
        "--kge_path", type=str, required=True, help="Path to the KGE model"
    )
    parser.add_argument(
        "--discriminator_folder", type=str, required=True, help="Discriminator module name"
    )
    parser.add_argument(
        "--root_dir", type=str, default=".", help="Root directory for imports"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Data type for the model (default: float32)",
    )
    parser.add_argument(
        "--exploration_weight", type=float, default=1.0, help="Exploration weight for MCTS"
    )
    parser.add_argument(
        "--leaf_threshold", type=int, default=10, help="Threshold for leaf node"
    )
    parser.add_argument(
        "--mcts_iterations", type=int, default=50, help="Number of MCTS iterations"
    )
    parser.add_argument(
        "--budget_per_entity", type=int, default=1000, help="Budget per sparse entity"
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, default=10, help="Save checkpoint every N entities"
    )
    args = parser.parse_args()

    # 添加上级目录到sys.path以导入自定义模块
    sys.path.append(args.discriminator_folder)
    sys.path.append(args.root_dir)

    # 延迟导入
    from kg_enhancer import KGEnhancer

    # 创建并运行Runner
    runner = Runner(args)
    runner.run()

    cleanup_distributed()
