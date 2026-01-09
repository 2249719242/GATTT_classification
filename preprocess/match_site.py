import pandas as pd
import os

# 定义文件路径
sub_label_path = '../sub_label.csv'
phenotypic_path = '../Phenotypic_V1.csv'

print(f"正在读取文件:\n1. {sub_label_path}\n2. {phenotypic_path}")

# 读取 CSV
df_sub = pd.read_csv(sub_label_path)
df_pheno = pd.read_csv(phenotypic_path)

print("Columns in sub_label.csv:", df_sub.columns.tolist())
print("Columns in Phenotypic_V1.csv:", df_pheno.columns.tolist())

# 统一 ID 列名以便合并
sub_id_col = 'subject' if 'subject' in df_sub.columns else 'SUB_ID'
# 遍历找一下可能的 ID 列
pheno_id_col = None
for col in ['SUB_ID', 'subject', 'Subject']:
    if col in df_pheno.columns:
        pheno_id_col = col
        break

if sub_id_col not in df_sub.columns:
    raise ValueError(f"在 sub_label.csv 中找不到 ID 列 (期望 'subject' 或 'SUB_ID')")
if not pheno_id_col:
    raise ValueError(f"在 Phenotypic_V1.csv 中找不到 ID 列")

print(f"合并主键: sub_label['{sub_id_col}'] <-> Phenotypic['{pheno_id_col}']")

# 确保 ID 类型一致 (可能是 int 或 str)
df_sub[sub_id_col] = df_sub[sub_id_col].astype(str).str.lstrip('0')
df_pheno[pheno_id_col] = df_pheno[pheno_id_col].astype(str).str.lstrip('0')

# 创建 ID -> SITE_ID 的映射字典
if 'SITE_ID' not in df_pheno.columns:
    raise ValueError("Phenotypic_V1.csv 中缺失 'SITE_ID' 列")

site_map = dict(zip(df_pheno[pheno_id_col], df_pheno['SITE_ID']))

# 映射 SITE_ID 到 df_sub
df_sub['SITE_ID'] = df_sub[sub_id_col].map(site_map)

# 检查匹配情况
missing_count = df_sub['SITE_ID'].isna().sum()
if missing_count > 0:
    print(f"警告: 有 {missing_count} 个被试没有找到对应的 SITE_ID。")
    # 如果没找到，可以用 'Unknown' 填充，或者检查原因
    df_sub['SITE_ID'].fillna('Unknown', inplace=True)
else:
    print("所有被试均成功匹配到 SITE_ID。")

# 保存结果
df_sub.to_csv(sub_label_path, index=False)
print(f"任务完成！已将 SITE_ID 添加并保存至: {sub_label_path}")
print("前5行预览:")
print(df_sub.head())
