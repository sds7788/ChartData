import os

def count_image_files(directory_path):
    """
    统计指定目录下的图片文件数量。

    Args:
        directory_path (str): 要检查的文件夹路径。

    Returns:
        int: 图片文件的总数。
        None: 如果路径无效或不是一个文件夹。
    """
    # 检查路径是否存在并且是一个文件夹
    if not os.path.isdir(directory_path):
        print(f"错误: '{directory_path}' 不是一个有效的文件夹路径。")
        return None

    # 定义常见的图片文件扩展名列表（可以根据需要添加更多）
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']
    
    image_count = 0
    
    try:
        # 遍历文件夹中的所有文件和子文件夹
        for item in os.listdir(directory_path):
            # 获取文件的完整路径
            full_path = os.path.join(directory_path, item)
            
            # 检查这是否是一个文件（而不是文件夹）
            if os.path.isfile(full_path):
                # 检查文件扩展名是否在我们的图片扩展名列表中
                # os.path.splitext(item)[1] 会获取文件的扩展名，例如 '.png'
                # .lower() 是为了确保无论大小写（如 .JPG 或 .jpg）都能匹配
                if os.path.splitext(item)[1].lower() in image_extensions:
                    image_count += 1
                    
    except OSError as e:
        print(f"读取文件夹时出错: {e}")
        return None

    return image_count

if __name__ == "__main__":
    # 提示用户输入文件夹路径
    folder_path = input("请输入您想统计图片数量的文件夹路径: ")
    
    # 调用函数并获取结果
    count = count_image_files(folder_path)
    
    # 如果函数成功返回了数量，就打印结果
    if count is not None:
        print(f"在文件夹 '{folder_path}' 中找到了 {count} 张图片。")

