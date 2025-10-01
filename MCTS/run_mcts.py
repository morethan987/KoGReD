from kg_enhancer import KGEnhancer
from utils import init_distributed, cleanup_distributed, shard_indices, get_device
from setup_logger import setup_logger
import torch
import torch_npu
import torch.distributed as dist
import os
import argparse
from tqdm.auto import tqdm


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.data_folder = args.data_folder
        self.logger = setup_logger(self.__class__.__name__)

        # åˆ�å§‹åŒ–åˆ†å¸ƒå¼�çŽ¯å¢ƒ
        self.is_initialed = init_distributed()

        # èŽ·å�–åˆ†å¸ƒå¼�ä¿¡æ�¯
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.device = get_device(self.local_rank)

        self.correct_relations = torch.load(
            f"{self.args.output_folder}/correct_relations.pth"
        )

        self.enhancer = KGEnhancer(
            entity2name_path=f"{self.data_folder}/entity2name.txt",
            relation2id_path=f"{self.data_folder}/relation2id.txt",
            entity2id_path=f"{self.data_folder}/entity2id.txt",
            entity2description_path=f"{self.data_folder}/entity2des.txt",
            kg_path=f"{self.data_folder}/train.txt",
            budget_per_entity=self.args.budget_per_entity,
            mcts_iterations=self.args.mcts_iterations,
            leaf_threshold=self.args.leaf_threshold,
            exploration_weight=self.args.exploration_weight,
            llm_path=self.args.llm_path,
            lora_path=self.args.lora_path,
            embedding_path=self.args.embedding_path,
            dtype=self.args.dtype,
            device=self.device)

        self.all_discovered_triplets = []

    def run(self):
        items = list(self.correct_relations.items())
        indices = shard_indices(
            len(items), self.world_size, self.rank)
        local_data = [items[i] for i in indices]

        progress = tqdm(
            total=len(local_data),
            desc=f"Processing (rank {self.rank})",
            disable=(self.rank != 0),
        )
        local_output = []
        for entity_idx, (entity, position_relations) in enumerate(local_data):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(
                f"Processing sparse entity {entity_idx + 1}/{len(items)}: {entity}"
            )
            self.logger.info(
                f"Position-relation pairs: {len(position_relations)}")
            # TODO: ç›®å‰�æ˜¯å¯¹æ¯�ä¸€ä¸ªä½�ç½®-å…³ç³»å¯¹å�•ç‹¬è¿›è¡ŒMCTSæ�œç´¢ï¼Œåº”è¯¥æŠŠå…³ç³»å¯¹çš„é€‰æ‹©ä¹Ÿæ”¾åˆ°MCTSä¸­
            for pos_rel_idx, (position, relation) in enumerate(position_relations):
                self.logger.info(f"\nProcessing pair {pos_rel_idx + 1}/{len(position_relations)}: "
                                 f"position={position}, relation={relation}"
                                 )

                # ä½¿ç”¨MCTSæ�œç´¢è¯¥å®žä½“-ä½�ç½®-å…³ç³»ç»„å�ˆçš„æ­£ç¡®ä¸‰å…ƒç»„
                discovered = self.enhancer.enhance_entity_relation(
                    entity, position, relation
                )

                local_output.extend(discovered)
                self.logger.info(
                    f"Discovered {len(discovered)} valid triplets for {entity}-{position}-{relation}")

        progress.close()

        # æ”¶é›†æ‰€æœ‰è¿›ç¨‹çš„ç»“æžœ
        if self.is_initialed:
            gathered = [None] * self.world_size if self.rank == 0 else None
            dist.all_gather_object(local_output, gathered, dest=0)
            if self.rank == 0:
                for triplet_list in gathered:
                    self.all_discovered_triplets.extend(triplet_list)
            else:
                return
        else:
            self.all_discovered_triplets = local_output

        # ä¿�å­˜æ‰€æœ‰å�‘çŽ°çš„ä¸‰å…ƒç»„
        output_path = os.path.join(
            self.args.output_folder, "discovered_triplets.txt"
        )
        os.makedirs(self.args.output_folder, exist_ok=True)
        self.logger.info(
            f"\nSaving {len(self.all_discovered_triplets)} discovered triplets to {output_path}"
        )
        with open(output_path, 'w', encoding='utf-8') as f:
            for head, rel, tail in self.all_discovered_triplets:
                f.write(f"({head}\t{rel}\t{tail})\n")

        self.logger.info("Knowledge graph enhancement completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCTS Runner")
    parser.add_argument(
        "--data_folder", type=str, required=True, help="Path to the dataset folder"
    )
    parser.add_argument(
        "--use_local_llm", action="store_true", help="Use local LLM instead of remote"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Data type for the model (default: float32)",
    )
    parser.add_argument(
        "--processed_data",
        type=str,
        default="MCTS/output/correct_relations.pth",
        help="Path to save/load preprocessed relations",
    )
    parser.add_argument(
        "--output_folder", type=str, default="MCTS/output", help="Output folder"
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
        "--llm_path", type=str, required=True, help="Path to the LLM model"
    )
    parser.add_argument(
        "--lora_path", type=str, required=True, help="Path to the LoRA weights"
    )
    parser.add_argument(
        "--embedding_path", type=str, required=True, help="Path to the kg embeddings"
    )
    parser.add_argument(
        "--kge_path", type=str, required=True, help="Path to the KGE model"
    )
    args = parser.parse_args()

    # åˆ›å»ºå¹¶è¿�è¡ŒRunner
    runner = Runner(args)
    runner.run()

    cleanup_distributed()
