#我现在需要读取id与标签，即读取Phenotypic_V1_0b_preprocessed1.csv的subject列与DX_GROUP列
import pandas as pd

# 读取CSV文件
df = pd.read_csv('Phenotypic_V1_0b_preprocessed1.csv')

# 提取subject列和DX_GROUP列
id_label_df = df[['subject', 'DX_GROUP']]

# 打印前几行数据
print(id_label_df.head())

# 保存到新的CSV文件
id_label_df.to_csv('id_label.csv', index=False)
print("已保存到id_label.csv")