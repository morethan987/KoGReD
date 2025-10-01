import os
import json
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from tqdm import tqdm
from peft import PeftModel
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist


base_path = '/home/ma-user/work/model/Alpaca-7B'

prompt_template = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Given a triple from a knowledge graph. Each triple consists of a head entity, a relation, and a tail entity. Please determine the correctness of the triple and response True or False.

### Input:
{}

### Response:

"""


def load_test_dataset(path):
    with open(path, "r") as f:
        test_dataset = json.load(f)
    return test_dataset


if __name__ == "__main__":
    # PyTorch 分布式训练需要'hccl'作为后端与华为NPU通信
    dist.init_process_group(backend='hccl')
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1)) # 获取总进程数
    device = f"npu:{local_rank}"
    torch.npu.set_device(device) # 绑定当前进程到指定NPU卡

    lora_weights = "/home/ma-user/work/moran/KoPA/output/alpaca_7b_fb_20250924_181238"
    test_data_path = "data/FB15K-237N-test.json"
    embedding_path = "{}/embeddings.pth".format(lora_weights)
    test_dataset = load_test_dataset(test_data_path)
    kg_embeddings = torch.load(embedding_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16
    ).to(device)
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.bfloat16,
    ).to(device)
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model = model.eval()
    result = []

    # 每个进程处理的数据量大约是总数除以world_size
    progress = tqdm(total=len(test_dataset) // world_size, disable=(local_rank != 0))

    # 数据并行切分，每个rank只处理自己的一部分数据
    for data in test_dataset[local_rank::world_size]:
        ent = data["input"]
        ans = data["output"]
        ids = data["embedding_ids"]
        ids = torch.LongTensor(ids).reshape(1, -1).to(device)
        prefix = kg_embeddings(ids).to(torch.bfloat16)
        prompt = prompt_template.format(ent)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        token_embeds = model.model.model.embed_tokens(
            input_ids).to(torch.bfloat16)
        input_embeds = torch.cat((prefix, token_embeds), dim=1)
        generate_ids = model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=16
        )
        context = tokenizer.batch_decode(
            input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = response.replace(context, "").strip()

        if local_rank == 0:
            print(response + '\n')

        result.append(
            {
                "answer": ans,
                "predict": response
            }
        )
        progress.update(1)
    progress.close()

    # 创建一个列表用于存放从各个进程收集来的结果
    gathered_results = [None] * world_size
    # 使用 all_gather_object 来收集 python 对象（这里是 list of dicts）
    dist.all_gather_object(gathered_results, result)

    # 只在主进程(rank 0)进行最终的指标计算和打印
    if local_rank == 0:
        # 将收集到的结果列表展开成一个大的列表
        final_result = [item for sublist in gathered_results for item in sublist]

        answer = []
        predict = []
        # 注意：这里我们使用 final_result，而不是原来的 result
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
            print(f"Total samples processed: {len(final_result)}")
            print(f"Accuracy: {acc}, Precision: {p}, Recall: {r}, F1-score: {f1}")
        else:
            print("No results to evaluate.")
