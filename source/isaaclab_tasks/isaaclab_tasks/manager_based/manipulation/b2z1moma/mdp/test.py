import torch

def distance_to_line(a, b, c):
    # 计算向量 ab 和 ac
    ab = b - a
    ac = c - a
    
    # 计算 ab 的模长的平方
    ab_norm_squared = torch.sum(ab**2, dim=1, keepdim=True)
    
    # 计算投影系数 t
    t = torch.sum(ab * ac, dim=1, keepdim=True) / ab_norm_squared
    
    # 计算投影点
    projection = a + t * ab
    
    # 计算 c 到投影点的距离
    distance = torch.norm(c - projection, dim=1)
    
    return distance

# 示例数据
a = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
b = torch.tensor([[1.0, 0.0], [2.0, 2.0]])
c = torch.tensor([[0.5, 0.5], [1.5, 1.5]])

# 计算距离
distances = distance_to_line(a, b, c)
print(distances.shape)