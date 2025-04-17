import os
import pandas as pd

# 创建目录并保存CSV文件
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# 读取数据
data = pd.read_csv(data_file)

# 分割输入特征和输出标签
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]

# 处理缺失值：数值列用均值填充，字符串列用众数或指定值填充
inputs['NumRooms'] = inputs['NumRooms'].fillna(inputs['NumRooms'].mean())
inputs['Alley'] = inputs['Alley'].fillna('Unknown')  # 或用 inputs['Alley'].mode()[0]

print("处理后的输入数据：")
print(inputs)
print("\n输出标签：")
print(outputs)