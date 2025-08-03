import os
import cv2
import numpy as np

# 输入和输出文件夹路径
input_folder = 'processed-cl'
output_folder = 'canny-edge-cl'

# 如果输出文件夹不存在，则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    # 检查文件是否为图像（简单检查扩展名）
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # 构建完整的文件路径
        input_path = os.path.join(input_folder, filename)

        # 读取图像为灰度图
        gray_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if gray_image is not None:
            # 应用Canny边缘检测
            blurred = cv2.GaussianBlur(gray_image, (5, 5), sigmaX=1.5)

            # Canny边缘检测
            low_threshold = 50  # 低阈值（弱边缘）
            high_threshold = 20  # 高阈值（强边缘）
            edges = cv2.Canny(blurred, low_threshold, high_threshold)


            # 构建输出路径
            output_path = os.path.join(output_folder, filename)

            # 保存处理后的图像
            cv2.imwrite(output_path, edges)
            print(f"Processed and saved: {output_path}")
        else:
            print(f"Failed to read image: {input_path}")

print("All images processed!")

def process_canny_images(folder_path, output_file):
    """处理canny-edge文件夹中的图像并保存二值化结果"""
    all_binary_data = []

    # 遍历文件夹中的所有文件
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            filepath = os.path.join(folder_path, filename)

            # 读取图像为灰度图
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"警告：无法读取图像 {filename}")
                continue

            print(f"处理 {filename} ({img.shape[1]}x{img.shape[0]})")


            # 二值化（>=128为1，否则为0）
            binary_img = (img >= 128).astype(np.uint8)

            # 展平为一维数组并添加到总数据中
            all_binary_data.extend(binary_img.flatten().tolist())
    # 保存到文本文件
    with open(output_file, 'w') as f:
        for value in all_binary_data:
            f.write(f"{value}\n")


folder_path = 'canny-edge-try'
output_file = 'canny-edge-try.txt'
process_canny_images(folder_path, output_file)