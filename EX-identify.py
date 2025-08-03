
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def read_and_combine_files(file10,file20):
    # 获取用户输入的文件名
    file1 = file10
    file2 = file20

    try:
        # 读取第一个文件
        with open(file1, 'r') as f1:
            lines1 = f1.readlines()

        # 读取第二个文件
        with open(file2, 'r') as f2:
            lines2 = f2.readlines()

        # 检查文件行数是否匹配
        if len(lines1) != len(lines2):
            raise ValueError("两个文件的行数不匹配")

        result = []
        for line1, line2 in zip(lines1, lines2):
            # 处理第一个文件的每一行
            a_values = [float(x.strip()) for x in line1.strip().split(',')]

            # 处理第二个文件的每一行
            b_value = float(line2.strip())

            # 合并数据并添加到结果中
            combined = a_values + [b_value]
            result.append(combined)

        return result

    except FileNotFoundError as e:
        print(f"错误：文件未找到 - {e}")
        return None
    except ValueError as e:
        print(f"错误：{e}")
        return None
    except Exception as e:
        print(f"发生未知错误：{e}")
        return None






import random


def reduce_dataset(dataset, a):
    # 设置随机种子
    random.seed(42)

    # 分离末尾为1和不为1的数组
    keep_list = [x for x in dataset if x[-1] == 1]
    print(len(keep_list))
    other_list = [x for x in dataset if x[-1] != 1]

    # 计算需要保留的其他数组的数量（向下取整）
    total_other = len(other_list)
    keep_other_count = int(a * total_other)

    # 随机选择要保留的其他数组
    kept_other = random.sample(other_list, keep_other_count)

    # 合并保留的数组
    reduced_dataset = keep_list + kept_other

    # 可以打乱顺序（可选）
    random.shuffle(reduced_dataset)

    return reduced_dataset





class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _gini(self, y):
        counter = Counter(y)
        gini = 1.0
        for label in counter:
            prob = counter[label] / len(y)
            gini -= prob ** 2
        return gini

    def _best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_gini = float('inf')

        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_indices = X[:, feature_idx] <= threshold
                right_indices = ~left_indices

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                gini_left = self._gini(y[left_indices])
                gini_right = self._gini(y[right_indices])

                weighted_gini = (len(y[left_indices]) * gini_left +
                                 len(y[right_indices]) * gini_right) / len(y)

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape

        # 终止条件
        if (depth == self.max_depth or
                num_samples < self.min_samples_split or
                len(np.unique(y)) == 1):
            return Counter(y).most_common(1)[0][0]

        # 找到最佳分割
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return Counter(y).most_common(1)[0][0]

        # 递归构建子树
        left_indices = X[:, feature] <= threshold
        right_indices = ~left_indices

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {'feature': feature, 'threshold': threshold,
                'left': left_subtree, 'right': right_subtree}

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _predict_sample(self, sample, tree):
        if isinstance(tree, dict):
            if sample[tree['feature']] <= tree['threshold']:
                return self._predict_sample(sample, tree['left'])
            else:
                return self._predict_sample(sample, tree['right'])
        else:
            return tree

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        n_features = X.shape[1]
        self.max_features = self.max_features or int(np.sqrt(n_features))

        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split)
            X_sample, y_sample = self._bootstrap_samples(X, y)

            # 特征采样
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
            X_sample = X_sample[:, feature_indices]

            tree.fit(X_sample, y_sample)
            self.trees.append((tree, feature_indices))
            print('n_estimator-',_)

    def predict(self, X):
        tree_preds = np.zeros((len(X), len(self.trees)))

        for i, (tree, feature_indices) in enumerate(self.trees):
            X_subset = X[:, feature_indices]
            tree_preds[:, i] = tree.predict(X_subset)

        # 多数投票
        return np.array([Counter(row).most_common(1)[0][0] for row in tree_preds])


def load_data(data):
    """将数据分割为特征和标签"""
    X = np.array([d[:-1] for d in data])
    y = np.array([d[-1] for d in data])
    return X, y


def evaluate(y_true, y_pred):
    """计算评估指标"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    return accuracy, precision, recall


def read_canny_edge_data(file_path):
    """
    读取canny边缘检测数据文件并将其保存到数组中

    参数:
    file_path (str): 数据文件的路径

    返回:
    numpy.ndarray: 包含数据的二维数组
    """
    try:
        # 读取文件数据到numpy数组
        data = np.loadtxt(file_path)
        return data
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None


def read_data_to_2d_array(file_path):
    """
    读取文件中的数据并转换为二维浮点数组

    参数:
        file_path (str): 要读取的文件路径

    返回:
        list: 二维浮点数组，格式如 [[a0,b0,c0,d0,...], [a1,b1,c1,d1,...], ...]
    """
    result = []
    with open(file_path, 'r') as file:
        for line in file:
            # 去除首尾空白字符和换行符
            stripped_line = line.strip()
            if stripped_line:  # 确保不是空行
                # 分割字符串并转换为浮点数
                row = [float(num) for num in stripped_line.split(',')]
                result.append(row)
    return result


# 示例数据集 - 最后一维是标签
# 这里使用一个简单的示例数据集，实际使用时可以替换为自己的数据
file10="features0.txt"
file20="groundtruth0.txt"
example_data = reduce_dataset(read_and_combine_files(file10,file20),0.025)
print(len(read_and_combine_files(file10,file20)))
print(len(example_data))
print(example_data[123])
print(example_data[2000])

# 加载数据
X, y = load_data(example_data)

# 创建随机森林模型
rf = RandomForest(n_estimators=11, max_depth=3)

# 训练模型
rf.fit(X, y)

file11="features0-test.txt"
file21="groundtruth0-test.txt"
canny_file="canny-edge-test.txt"
X_test=read_data_to_2d_array(file11)
y_test=read_canny_edge_data(file21)
y_canny = read_canny_edge_data(canny_file)
y_pred = rf.predict(np.array(X_test))
print(X_test[2])
if len(y_pred)==269664:
    for i in range(636*2, len(y_pred)-636*2):
        if y_canny[i] == 1 :
            y_pred[i] = y_pred[i + 1] = y_pred[i - 1] = y_pred[i + 636] = y_pred[i - 636] = y_pred[i - 636*2] = y_pred[i + 636*2] = y_pred[i - 2] = y_pred[i + 2] = y_pred[i - 636-1] = y_pred[i - 636+1] = y_pred[i + 636 +1] = y_pred[i + 636 -1 ]= y_pred[i + 636 +2] =y_pred[i + 636 -2 ] = y_pred[i - 636 +2]  = y_pred[i - 636 -2 ] = y_pred[i - 636*2-1] = y_pred[i + 636*2-1] = y_pred[i - 636*2+1]  = y_pred[i + 636*2+1] = 0
if len(y_pred)==174080:
    for i in range(512*2, len(y_pred)-512*2):
        if y_canny[i] == 1 :
            y_pred[i] = y_pred[i + 1] = y_pred[i - 1] = y_pred[i + 512] =y_pred[i - 512] = y_pred[i - 512*2] = y_pred[i + 512*2] = y_pred[i - 2] = y_pred[i + 2] = y_pred[i - 512-1] = y_pred[i - 512+1] =y_pred[i + 512 +1] = y_pred[i + 512 -1 ] = y_pred[i + 512 +2] = y_pred[i + 512 -2 ] = y_pred[i - 512 +2] = y_pred[i - 512 -2 ] = y_pred[i - 512*2-1] = y_pred[i + 512*2-1] = y_pred[i - 512*2+1] = y_pred[i + 512*2+1] =0
if len(y_pred)==86400:
    for i in range(360*2, len(y_pred)-360*2):
        if y_canny[i] == 1 :
            y_pred[i] = y_pred[i + 1] = y_pred[i - 1] =y_pred[i + 360] = y_pred[i - 360] = y_pred[i - 360*2] = y_pred[i + 360*2] = y_pred[i - 2] = y_pred[i + 2] = y_pred[i - 360-1] = y_pred[i - 360+1] = y_pred[i + 360 +1] = y_pred[i + 360 -1 ] =y_pred[i + 360 +2] = y_pred[i + 360 -2 ] = y_pred[i - 360 +2] = y_pred[i - 360 -2 ] = y_pred[i - 360*2-1] = y_pred[i + 360*2-1] = y_pred[i - 360*2+1] = y_pred[i + 360*2+1] =0
if len(y_pred)==94000:
    for i in range(376*2, len(y_pred)-376*2):
        if y_canny[i] == 1 :
            y_pred[i] =  y_pred[i + 1] = y_pred[i - 1] = y_pred[i + 376] = y_pred[i - 376] = y_pred[i - 376*2] = y_pred[i + 376*2] = y_pred[i - 2] = y_pred[i + 2] =y_pred[i - 376-1] = y_pred[i - 376+1] = y_pred[i + 376 +1] = y_pred[i + 376 -1 ] =y_pred[i + 376 +2] = y_pred[i + 376 -2 ] = y_pred[i - 376 +2] = y_pred[i - 376 -2 ] = y_pred[i - 376*2-1] = y_pred[i + 376*2-1] =y_pred[i - 376*2+1] = y_pred[i + 376*2+1] =0

print("\nEvaluation on test data:")
evaluate(y_test, y_pred)





# 示例数据 - 替换为你的真实数据
y_true = y_test


# 计算指标
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

# 打印指标
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print("混淆矩阵:")
print(cm)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

# 添加标签
classes = ['Negative', 'Positive']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# 在矩阵中显示数值
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# 可视化指标对比
metrics = ['Accuracy', 'Precision', 'Recall']
values = [accuracy, precision, recall]

plt.figure(figsize=(8, 5))
plt.bar(metrics, values, color=['blue', 'green', 'orange'])
plt.ylim(0, 1.1)
plt.title('Classification Metrics Comparison')
plt.ylabel('Score')

# 在柱子上显示数值
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.4f}", ha='center')

plt.show()






# 新数据预测
# 读取文件并将数据转换为整数数组
def read_features_file(file_path):
    features_array = []
    with open(file_path, 'r') as file:
        for line in file:
            # 去除行尾的换行符并按逗号分割
            numbers = line.strip().split(',')
            # 将字符串转换为整数并添加到当前行列表
            int_numbers = [float(num) for num in numbers]
            features_array.append(int_numbers)
    return features_array

# 文件路径
file_path = 'features0-try.txt'
canny_file = 'canny-edge-try.txt'
new_data = np.array(read_features_file(file_path))
canny_info = read_canny_edge_data(canny_file)
predictions = rf.predict(new_data)
a=0
for i in range(len(predictions)):
    if predictions[i] ==1:
        a+=1
print("白点数(前)：",a)

for i in range(636, len(predictions) - 636):
    if y_canny[i] == 1 :
        predictions[i] = predictions[i + 1] = predictions[i - 1] = predictions[i + 636] = predictions[i - 636] = predictions[i - 636*2] = predictions[i + 636*2] = predictions[i - 2] = predictions[i + 2] = predictions[i - 636-1] = predictions[i - 636+1] = predictions[i + 636 +1] = predictions[i + 636 -1 ]= predictions[i + 636 +2] =predictions[i + 636 -2 ] = predictions[i - 636 +2]  = predictions[i - 636 -2 ] = predictions[i - 636*2-1] = predictions[i + 636*2-1] = predictions[i - 636*2+1]  = predictions[i + 636*2+1] = 0

a=0
for i in range(len(predictions)):
    if predictions[i] ==1:
        a+=1
print("白点数(后)：",a)


# 假设你的数据存储在一个名为'data'的列表中，格式为[0.0, 1.0, ...]共269664个值
# 这里我们创建一个示例数据（实际使用时替换为你的真实数据）
data = predictions  # 示例随机数据，实际使用时替换为你的真实数据

# 确保数据长度正确
assert len(data) == 636 * 424, "数据长度与图像尺寸不匹配"

# 将数据转换为numpy数组并重塑为636x424的图像
image_data = np.array(data, dtype=np.float32).reshape((424, 636))  # 注意OpenCV使用高度x宽度

# 确保输出目录存在
output_dir = 'gt'
os.makedirs(output_dir, exist_ok=True)

# 保存图像
output_path = os.path.join(output_dir, 'output_image10.png')
cv2.imwrite(output_path, image_data * 255)  # 将0-1范围转换为0-255范围

print(f"图像已成功保存到: {output_path}")