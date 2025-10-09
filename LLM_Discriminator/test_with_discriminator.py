import os
import argparse
import json
import torch
import torch.distributed as dist
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_test_dataset(path):
    with open(path, "r") as f:
        test_dataset = json.load(f)
    return test_dataset

def main(args):
    # 1. 初始化分布式环境
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    # 2. 实例化判别器类
    # embedding_path的构造逻辑与原脚本保持一致
    embedding_path = "{}/embeddings.pth".format(args.lora_weights)
    discriminator = TriplesDiscriminator(
        llm_path=args.base_model,
        lora_path=args.lora_weights,
        embedding_path=embedding_path,
        device=device,
        batch_size=args.batch_size
    )

    # 3. 加载并切分数据集
    if local_rank == 0:
        print(f"Loading test data from {args.test_data}...")
    test_dataset = load_test_dataset(args.test_data)
    # 每个进程处理自己的数据子集
    local_data_slice = test_dataset[local_rank::world_size]

    if local_rank == 0:
        print(f"Starting prediction on {world_size} devices...")

    # 4. 使用判别器进行推理
    predictions = discriminator.judge_batch(local_data_slice)

    # 5. 准备待聚合的结果
    # 将预测结果与真实标签配对，以便后续统一计算指标
    results_to_gather = []
    for i, pred_result in enumerate(predictions):
        ground_truth_output = local_data_slice[i]["output"]
        predicted_output = "True" if pred_result["is_correct"] else "False"
        results_to_gather.append({
            "answer": ground_truth_output,
            "predict": predicted_output
        })

    # 6. 从所有进程收集结果
    gathered_results = [None] * world_size
    dist.barrier()  # 确保所有进程都到达此点
    dist.all_gather_object(gathered_results, results_to_gather)

    # 7. 仅在主进程(rank 0)上计算和打印最终指标
    if local_rank == 0:
        print("\nAll ranks finished. Calculating metrics on rank 0...")
        final_result = [item for sublist in gathered_results for item in sublist]

        answer = []
        predict = []
        for data in final_result:
            if "True" in data["answer"]:
                answer.append(1)
            else:
                answer.append(0)
            if "True" in data["predict"]:
                predict.append(1)
            else:
                predict.append(0)

        # 确保有预测结果，避免除零错误
        if len(answer) > 0 and len(predict) > 0:
            acc = accuracy_score(y_true=answer, y_pred=predict)
            p = precision_score(y_true=answer, y_pred=predict)
            r = recall_score(y_true=answer, y_pred=predict)
            f1 = f1_score(y_true=answer, y_pred=predict)
            print("=========================================================")
            print(f"Total samples processed: {len(final_result)}")
            print(f"Accuracy: {acc:.4f}, Precision: {p:.4f}, Recall: {r:.4f}, F1-score: {f1:.4f}")
            print("=========================================================")
        else:
            print("No results to evaluate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="Path to the base model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--lora_weights", type=str, required=True, help="Path to the LoRA weights")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for the discriminator inference")
    parser.add_argument("--root_dir", type=str, default=".", help="Root directory for saving outputs")
    args = parser.parse_args()

    import sys
    sys.path.append(args.root_dir)
    from discriminator import TriplesDiscriminator

    main(args)
