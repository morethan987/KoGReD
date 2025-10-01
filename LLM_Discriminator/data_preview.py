import torch


def preview_kge(file_path: str):
    """预览KGE模型的pth文件"""
    data = torch.load(file_path, map_location='cpu')
    print("Keys in the loaded data:\n", data.keys())
    print()


if __name__ == "__main__":
    main()
