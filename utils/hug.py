from huggingface_hub import HfApi
api = HfApi()

# 路径修正建议：您的路径混合了 \\ 和 \，在使用 r"..." 时，建议统一为单个 \
folder_path = r"D:\桌面\数据存储\dataset(1,20k)"

# allow_patterns 列表需要手动区分文件和文件夹
allow_patterns = [
    "images_structured/**",      # <-- 对于文件夹，使用 /**
    "train_dataset.json",        # <-- 对于文件，直接写文件名
    "sampled_dataset_2000.json",  # <-- 对于文件，直接写文件名
    "images.zip"
]

print(f"准备上传，只包含模式: {allow_patterns}")

api.upload_large_folder(
    folder_path=folder_path,
    repo_id="sds77/ChartData",
    repo_type="dataset",
    allow_patterns=allow_patterns
)

print("✅ 上传脚本执行完毕。")