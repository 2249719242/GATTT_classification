import numpy as np
base_dir='./filt_global'
# 遍历文件夹下的所有文件

import os
import pandas as pd

# 创建一个列表来存储文件名
all_data = []
file_records = []

print("正在处理文件...")
for file in sorted(os.listdir(base_dir)):
    file_path = os.path.join(base_dir, file)
    if os.path.isfile(file_path):
        # 加载文件名
        file_name = os.path.splitext(file)[0]
        # 存储文件名到 file_records 列表
        file_records.append(file_name)
        print(file_name)

# 打印存储的文件名数量和前几个文件名
print(f"共存储了 {len(file_records)} 个文件名")
print("前5个文件名:")
for i in range(min(5, len(file_records))):
    print(file_records[i])



#我需要更改file_records中的文件名，需要文件名的第一个_与第二个_之间的数字（同时去除前面的0），比如文件名为：Caltech_0051461_rois_cc200,我需要51461这个数字
# 从文件名中提取第一个_与第二个_之间的数字（去除前面的0）
#创建一个列表，用于存储提取的ID
id_list = []
for file_name in file_records:
    # 使用split('_')分割文件名
    parts = file_name.split('_')
    if len(parts) >= 2:
        # 获取第一个_与第二个_之间的部分，并去除前导零，如果该部分为a、b、c、d、1、2其中的一个，则获取第二个_与第三个_之间的部分
        subject_id = parts[2].lstrip('0') if parts[1] in ['a', 'b', 'c', 'd', '1', '2'] and len(parts) >= 3 else parts[1].lstrip('0')
        id_list.append(subject_id)
# 打印id_list前五个
print("id_list前五个:")
for i, id in enumerate(id_list):
    if i < 5:
        print(id)
    else:
        break
#将id_list输出为csv文件的一列
import csv
with open('filename_list.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for id in id_list:
        writer.writerow([id])
