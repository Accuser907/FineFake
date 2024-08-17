import numpy as np
from torchmetrics import ConfusionMatrix
import torch
# # 示例数组
# array1 = np.array([0, 1, 0, 1, 0, 1])  # 1 的索引为 1, 3, 5
# array2 = np.array([10, 20, 30, 40, 50, 60])

# # 找到值为1的索引
# indices = np.where(array1 == 1)

# # 根据索引获取另一个数组的元素
# new_array = array2[indices]

# # 打印结果
# print("Indices with value 1:", indices)
# print("Corresponding elements from array2:", new_array)
a = ConfusionMatrix(task="binary")
predictions = torch.tensor([0,1,0,1])
targets = torch.tensor([0,1,1,1])
a.update(preds = predictions,target=targets)
output = a.compute()
print(output)