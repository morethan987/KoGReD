import os
import torch.distributed as dist
import torch
import torch_npu
from datetime import timedelta
from setup_logger import logging


logger = logging.getLogger(__name__)


def get_sparse_entities(dataset, sparse_threshold: float = 1.1e-5):
    """返回稀疏实体列表"""
    count = {}
    with open(f"{dataset}/train.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()
        triple_num = len(lines)
        for line in lines:
            head, rel, tail = line.strip().split("\t")  # 修正顺序
            count[head] = count.get(head, 0) + 1
            count[tail] = count.get(tail, 0) + 1

    sparse_ent = []
    for entity, num in count.items():
        # 统一比例制，不乘 100
        frequency = num / (triple_num * 2)
        if frequency < sparse_threshold:
            sparse_ent.append(entity)

    logger.info(
        f"稀疏实体列表计算成功! 从{triple_num}个三元组中发现{len(sparse_ent)}个稀疏结点, "
        f"占比{round(len(sparse_ent) / len(count) * 100, 4)}%, 筛选阈值为: {sparse_threshold}"
    )
    return sparse_ent


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def ddp_setup():
    dist.init_process_group(backend="hccl", timeout=timedelta(seconds=1800))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.npu.set_device(local_rank)
    return rank, local_rank, world_size


def shard_indices(total_size: int, rank: int, world_size: int):
    """数据分片函数"""
    per_shard = total_size // world_size
    remainder = total_size % world_size

    start_idx = rank * per_shard + min(rank, remainder)
    end_idx = start_idx + per_shard + (1 if rank < remainder else 0)

    return list(range(start_idx, end_idx))


def get_device(local_rank: int):
    if is_distributed():
        if torch.npu.is_available():
            device = torch.device(f'npu:{local_rank}')
        elif torch.cuda.is_available():
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cpu')
    else:
        if torch.npu.is_available():
            device = torch.device('npu:0')
        elif torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
    return device


def init_distributed():
    """初始化单机多卡分布式环境"""
    if is_distributed():
        rank, local_rank, world_size = ddp_setup()

        # 根据设备类型选择后端
        if torch.npu.is_available():
            backend = 'hccl'  # 华为NPU使用hccl后端
        elif torch.cuda.is_available():
            backend = 'nccl'  # NVIDIA GPU使用nccl后端
        else:
            backend = 'gloo'  # CPU使用gloo后端

        try:
            # 初始化进程组
            dist.init_process_group(
                backend=backend,
                world_size=world_size,
                rank=rank
            )
            dist.barrier()  # 同步所有进程

            if rank == 0:
                logger.info(
                    f"Initialized single-node distributed training with backend: {backend}")
                logger.info(
                    f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")

            return True
        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e}")
            raise e
    return False


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    dataset = "data/FB15K-237N"
    sparse_entities = get_sparse_entities(dataset)
