import os

def list_folders_with_index(directory, output_file):
    # 获取指定目录中的所有文件夹
    folders = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
    # 对文件夹进行排序
    folders.sort()

    # 打开输出文件进行写入
    with open(output_file, 'w') as f:
        for index, folder in enumerate(folders):
            f.write(fr".\LoRA_models\{folder}"+"\n")

    print(f"Folder list saved to {output_file}")

if __name__ == "__main__":
    # 获取当前工作目录
    current_directory = os.getcwd()
    
    # 定义输出文件名
    output_file = 'folders_list.txt'
    
    # 列出当前目录中的所有文件夹，并将结果保存到文件中
    list_folders_with_index(current_directory, output_file)
