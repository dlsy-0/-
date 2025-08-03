#导入数据集，预处理，分割数据集，分类预测，评估

import cv2
import os
import numpy as np

def Process_Image(image):
    image = cv2.imread(image)
    rs_img = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    rs_img = cv2.resize(rs_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    # 应用自适应直方图均衡化
    b, g, r = cv2.split(rs_img)
    return g

def Save_Image(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in sorted(os.listdir('image')):
        # 检查文件是否是图片（简单检查扩展名）
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 构建完整的输入文件路径
            input_path = os.path.join('image', filename)

            # 读取图片

            processed_image = Process_Image(input_path)

            # 构建输出文件路径
            output_path = os.path.join(output_folder, filename)

            # 保存处理后的图片
            cv2.imwrite(output_path, processed_image)
            print(f"已处理并保存: {output_path}")





def get_pixel_features(img, x, y, height, width):
    """获取单个像素点的5个特征"""
    features = []
    pixel_value = float(img[y, x])  # 转为 float 避免溢出

    # 1. 像素强度
    features.append(pixel_value)

    # 2. 像素强度与图像平均强度的比值
    mean_intensity = np.mean(img)
    if mean_intensity != 0:
        ratio = pixel_value*100 / mean_intensity
    else:
        ratio = 0
    features.append(ratio)

    # 3-5. 与不同邻域平均强度的差
    for radius in [1,2]:  # 1: 3x3 (8邻域), 2: 5x5 (24邻域), 3: 7x7 (48邻域)
        # 提取邻域
        x_min = max(0, x - radius)
        x_max = min(width, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(height, y + radius + 1)

        neighborhood = img[y_min:y_max, x_min:x_max]

        # 排除中心点（如果存在）
        center_rel_x = x - x_min
        center_rel_y = y - y_min
        if (0 <= center_rel_x < neighborhood.shape[1] and
                0 <= center_rel_y < neighborhood.shape[0]):
            # 使用 mask 排除中心点
            mask = np.ones_like(neighborhood, dtype=bool)
            mask[center_rel_y, center_rel_x] = False
            neighborhood = neighborhood[mask]

        # 计算邻域平均值（空邻域视为0）
        if neighborhood.size > 0:
            neighborhood_avg = np.mean(neighborhood)
        else:
            neighborhood_avg = 0.0

        # 计算与中心像素的差
        diff = pixel_value - neighborhood_avg
        features.append(neighborhood_avg)
        features.append(diff)

    return features


def process_images_in_folder(folder_path):
    """处理文件夹中的所有图像"""
    all_features = []

    for filename in sorted(os.listdir(folder_path)):
        print("file:",filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            filepath = os.path.join(folder_path, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            height, width = img.shape
            print(f"Processing {filename} ({width}x{height})")

            # 转换为 float32 避免溢出
            img_float = img.astype(np.float32)

            for y in range(height):
                for x in range(width):
                    features = get_pixel_features(img_float, x, y, height, width)
                    all_features.append(features)

    return np.array(all_features)




def process_groundtruth_images(folder_path, output_file):
    """处理groundtruth文件夹中的图像并保存二值化结果"""
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

            # 缩小至1/4大小
            small_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            small_img = cv2.resize(small_img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)


            # 二值化（>=128为1，否则为0）
            binary_img = (small_img >= 128).astype(np.uint8)

            # 展平为一维数组并添加到总数据中
            all_binary_data.extend(binary_img.flatten().tolist())

    # 保存到文本文件
    with open(output_file, 'w') as f:
        for value in all_binary_data:
            f.write(f"{value}\n")

    print(f"二值化数据已保存至 {output_file}")
    print(f"总像素数: {len(all_binary_data)}")


# 使用示例
folder_path = 'groundtruth'
output_file = 'groundtruth0.txt'
process_groundtruth_images(folder_path, output_file)


def Get_data():
    return 0



output_folder = 'processed'  # 输出文件夹
Save_Image(output_folder)

input_folder = 'processed'
features_array = process_images_in_folder(input_folder)

# 打印结果摘要
print(f"总共提取了 {features_array.shape[0]} 个像素的特征")
print(f"每个特征向量的维度: {features_array.shape[1]}")
print(f"特征数组形状: {features_array.shape}")  # (N, 5)
print(features_array[133094])
np.savetxt('features0.txt',features_array, fmt='%.6f', delimiter=',')


