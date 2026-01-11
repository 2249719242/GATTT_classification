````markdown
# GATTT：双流时空-几何网络（含 DANN）用于 rs-fMRI 分类

本仓库包含论文/实验中使用的 **GATTT** 模型代码与复现实验脚本，支持主模型训练与多种消融实验。

---

## 1. 环境配置

- 操作系统：Ubuntu 22.04  
- Python：3.12  
- PyTorch：2.5.1  
- CUDA：12.4  
- 训练平台：AutoDL  
---

## 2. 项目结构（简要）

- `GATTT/`：主代码目录（训练与模型相关）
- `GATTT/preprocess/`：数据下载与预处理脚本目录

---

## 3. 数据下载与预处理

### 3.1 下载 fMRI 数据
进入预处理目录：

```bash
cd GATTT/preprocess
````

使用 `download` 脚本下载 fMRI 文件（按项目脚本说明执行）：

```bash
# 示例：根据你的 download 脚本实际用法运行
python download.py
```

### 3.2 预处理数据

下载并处理好原始文件后，运行预处理脚本：

```bash
python Preprocess.py
```

> 说明：该步骤会将原始数据转换为训练所需的输入格式（如 ROI 时间序列、图结构/特征等），并生成后续训练所依赖的数据文件。

---

## 4. 训练与复现实验

回到主目录：

```bash
cd ../
# 或者从项目根目录直接：
# cd GATTT
```

### 4.1 主模型训练（GAT-Trans + Geometry + DANN）

运行以下命令获得主模型效果：

```bash
python GATTTtrain.py
```

---

## 5. 消融实验（Ablation Studies）

### 5.1 去掉站点对抗（w/o DANN）

关闭 DANN 判别器/站点对抗模块：

```bash
python GATTTtrain.py --no_dann
```

### 5.2 去掉几何流（w/o Geometry）

关闭几何流分支（Geometry Stream）：

```bash
python GATTTtrain.py --no_geometry
```

### 5.3 仅 GAT / Transformer（GAT_only / Trans_only）

运行仅 GAT / Transformer 的训练脚本：

```bash
python train_GAT__Trans_Only.py
```

---

## 6. 备注

* 请先完成 **数据下载与预处理**，再运行训练脚本，否则会因缺少输入数据而报错。

---

如有问题欢迎提 Issue 或联系作者。

```
```
