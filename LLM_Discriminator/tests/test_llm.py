import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ====== 全局配置 ======
MODEL_PATH = "wxjiao/alpaca-7b"
# =================================================

def main():
    print(f"正在加载模型: {MODEL_PATH}")

    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)

    # 自动选择设备
    device = "cuda:2"
    model.to(device)
    print(f"模型已加载到设备: {device}")

    # 测试输入
    prompt = "你好，介绍一下你自己。"
    print(f"\n输入提示: '{prompt}'")

    # 编码并生成
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.9)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"生成结果: '{result}'")
    print("\n模型测试成功！")

if __name__ == "__main__":
    # cdko && ackopa && python LLM_Discriminator/tests/test_llm.py
    main()
