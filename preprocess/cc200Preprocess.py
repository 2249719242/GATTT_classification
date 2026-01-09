import numpy as np
base_dir='../filt_global'
# 遍历文件夹下的所有文件
output_dir = './data' # 用于存放处理后数据的文件夹

import os
import pandas as pd

# 创建一个列表来存储所有裁剪后的数据
all_data = []
file_records = []
file_names=[]
all_processed_data = [] # 用于存储所有处理好的数据
print("正在处理文件并裁剪...")
def augment_fmri_sliding_window(fmri_data, window_size=80, stride=20):
    """
    使用滑动窗口对单个被试的fMRI数据进行增强。
    """
    current_length = fmri_data.shape[0]
    windows = []
    
    if current_length < window_size:
        # 如果长度不足，先进行对称填充到 window_size
        pad_width = window_size - current_length
        padded_data = np.pad(fmri_data, ((0, pad_width), (0, 0)), mode='symmetric')
        windows.append(padded_data)
    else:
        # 滑动窗口提取
        for start in range(0, current_length - window_size + 1, stride):
            end = start + window_size
            windows.append(fmri_data[start:end, :])
            
    return windows

# 存储所有样本、标签、归属 ID 和 站点 ID
all_samples = []
all_labels_augmented = []
all_subject_ids = []
all_site_ids = []

# 加载标签文件
label_df = pd.read_csv('../sub_label.csv')

# 建立映射:
# ID -> DX_GROUP (Diagnosis)
# ID -> SITE_ID (Site)
label_map = dict(zip(label_df['subject'].astype(str), label_df['DX_GROUP']))
site_map_raw = dict(zip(label_df['subject'].astype(str), label_df['SITE_ID']))

# 将站点名称转换为数字 ID (Label Encoding)
from sklearn.preprocessing import LabelEncoder
site_encoder = LabelEncoder()
# 获取所有唯一的站点名称并拟合
unique_sites = label_df['SITE_ID'].dropna().unique()
site_encoder.fit(unique_sites)
print(f"检测到 {len(unique_sites)} 个站点: {unique_sites}")

print("正在处理文件并应用滑动窗口增强...")

window_size = 80
stride = 20

processed_files_count = 0
skipped_files_count = 0

for file in sorted(os.listdir(base_dir)):
    file_path = os.path.join(base_dir, file)
    if os.path.isfile(file_path):
        # 提取 Subject ID
        file_name = os.path.splitext(file)[0]
        parts = file_name.split('_')
        if len(parts) >= 2:
            sub_id_raw = parts[2].lstrip('0') if parts[1] in ['a', 'b', 'c', 'd', '1', '2'] and len(parts) >= 3 else parts[1].lstrip('0')
            
            if sub_id_raw in label_map and sub_id_raw in site_map_raw:
                # 获取该被试的 Diagnosis 和 Site
                diag_label = label_map[sub_id_raw]
                site_name = site_map_raw[sub_id_raw]
                site_label = site_encoder.transform([site_name])[0]
                
                # 加载数据
                data = np.loadtxt(file_path, comments='#')
                data = (data - np.mean(data)) / (np.std(data) + 1e-8)
                
                windows = augment_fmri_sliding_window(data, window_size=window_size, stride=stride)
                
                for win in windows:
                    all_samples.append(win)
                    all_labels_augmented.append(diag_label)
                    all_subject_ids.append(sub_id_raw)
                    all_site_ids.append(site_label)
                
                processed_files_count += 1
            else:
                # print(f"跳过: {file} (未找到标签或站点信息)")
                skipped_files_count += 1

# 转换为 numpy 数组
X = np.array(all_samples)
y = np.array(all_labels_augmented)
groups = np.array(all_subject_ids)
sites = np.array(all_site_ids)

print(f"\n处理完成!")
print(f"总被试: {processed_files_count}, 样本数: {X.shape[0]}")
print(f"站点数量: {len(unique_sites)}")

# 保存
np.save('X_augmented.npy', X)
np.save('y_augmented.npy', y)
np.save('groups_augmented.npy', groups)
np.save('sites_augmented.npy', sites) # 新增：保存站点标签
print("数据已保存: X, y, groups, sites")
