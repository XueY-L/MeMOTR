import os
from imagecorruptions import corrupt
import matplotlib.pyplot as plt
from imagecorruptions import get_corruption_names


def process_image(image_path, output_path, cor_name, severity):
    """
    对图像进行处理并保存到指定路径。
    :param image_path: 输入图像路径
    :param output_path: 输出图像路径
    """
    img = plt.imread(image_path)
    corrupted_image = corrupt(img, corruption_name=cor_name, severity=severity)
    plt.imsave(output_path, corrupted_image)


def process_all_images(input_dir, output_dir, cor_name, severity):
    """
    递归遍历输入目录中的所有jpg文件，处理后按相同结构保存到输出目录。
    :param input_dir: 输入根目录
    :param output_dir: 输出根目录
    """
    for subfolder in os.listdir(input_dir):
        subfolder_path = os.path.join(input_dir, subfolder)
        if os.path.isdir(subfolder_path):
            # 为每个子文件夹创建对应的输出目录
            output_subfolder = os.path.join(output_dir, subfolder)
            os.makedirs(output_subfolder, exist_ok=True)

            print(f"Processing folder: {subfolder_path}")
            # 遍历子文件夹中的 jpg 文件
            for file in os.listdir(subfolder_path):
                if file.lower().endswith(".jpg"):
                    input_path = os.path.join(subfolder_path, file)
                    output_path = os.path.join(output_subfolder, file)
                    process_image(input_path, output_path, cor_name, severity)

for c in ['brightness', 'contrast', 'defocus_blur', 'fog', 'frost', 'gaussian_blur', 'jpeg_compression', 'motion_blur', 'saturate', 'spatter']:
    #  ['brightness', 'contrast', 'defocus_blur', 'fog', 'frost', 'gaussian_blur', 'jpeg_compression', 'motion_blur', 'saturate', 'spatter']
    severity = 5
    input_dir = '/root/BDD100K/images/track/val'
    # output_dir = f'/workspace/BDD100K/{c}-{severity}'
    output_dir = f'/root/BDD100K/images/track/val-corurupt/{c}-{severity}'
    process_all_images(input_dir, output_dir, c, severity)