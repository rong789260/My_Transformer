# My_Transformer
基于 Transformer 架构的英文到中文翻译工具，包含完整的模型训练、评估及推理功能，并支持消融实验消融实验分析关键组件（如位置编码）的作用。


## 项目简介

本项目实现了一个轻量级 Transformer 模型，用于英文到中文的平行语料翻译。主要功能包括：
- 数据预处理（文本清洗、词汇表构建、数据集封装）
- Transformer 模型训练与评估（支持损失曲线、BLEU 分数可视化）
- 消融实验（对比有无位置编码对模型性能的影响）
- 翻译推理（加载训练好的模型进行实时英中翻译）


## 文件夹结构
transformer-translation/
├── data/ # 数据集
├── models/ # 训练好的模型权重
├── results/ # 实验结果（损失曲线、BLEU 曲线等）
├── src/ # 核心代码
│ ├── train.py # Baseline模型与消融实验
│ ├── com_train.py # 对比实验
│ └── translate.py # 翻译推理脚本
├── requirements.txt # 依赖清单
└── README.md # 项目说明

## 环境配置

### 依赖项
项目依赖以下 Python 库（详见 `requirements.txt`）：
- Python 3.8 
- torch == 2.4.1（深度学习框架）
- numpy == 1.24.1（数值计算）
- nltk == 3.9.1（BLEU 评分计算）
- matplotlib == 3.7.5（可视化）

### 安装步骤
1. 克隆项目到本地（或下载源码）：
   ```bash
   git clone <https://github.com/rong789260/My_Transformer>
   cd My_Transformer
2. 创建虚拟环境：
   # 使用conda
   conda create -n transformer python=3.8 -y
   conda activate transformer
3. 安装依赖：
   pip install -r requirements.txt

## 使用方法
1. 准备数据集
数据集格式：每行一条数据，英文和中文用 \t 分隔（如 Hello world!\t你好，世界！）
需在 data/目录下放置三个文件：
train.txt：训练集
dev.txt：验证集
test.txt：测试集
2. 模型训练与实验
Baseline模型与消融实验
python src/train.py
训练参数可在 train.py 中修改（如 batch_size=8、epochs=100 等）
对比实验
python src/com_train.py
训练过程中会自动保存模型权重到 models/ 目录
实验结果（损失曲线、BLEU 曲线）会保存到 results/ 目录
3. 翻译推理
python src/translate.py
程序会自动加载 models/baseline_model.pt（基准模型）
支持交互式输入英文句子，输出翻译结果（输入 q 退出）

## 实验说明
### 消融实验设计
项目包含两组对比实验：
Baseline：完整 Transformer 模型（含位置编码和层归一化）
Without Positional Encoding：移除位置编码的模型
### 对比实验设计
项目包含三组对比实验：
不同的注意力头数（2，4，8）、不同的学习率（1e-4，1e-5，5e-4）、不同的batch size（4，8，16）
### 评估指标
损失函数：交叉熵损失（忽略 <PAD> 符号）
翻译质量：BLEU 分数（使用 nltk.translate.bleu_score 计算）
### 实验结果
损失曲线和 BLEU 曲线保存在 results/ 目录
测试集最终结果会在训练结束后打印（包含损失、BLEU 分数和训练时间）
### 注意事项
若使用 GPU 加速，需确保 PyTorch 版本与 CUDA 兼容（推荐 CUDA 11.6+）
数据集规模较小时，可减小 batch_size 或模型参数（如 d_model=64）避免过拟合
长句子翻译可能需要调整 max_seq_len 参数（默认 20）
