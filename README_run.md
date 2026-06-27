# BiRNA-BERT NUC Baseline

第一版目标：

```text
BiRNA-BERT + NUC tokenization + mean pooling + center pooling + MLP classifier
```

基础 baseline 不包含 BPE、FiLM、MoE、双视角融合；当前脚本已支持可选 LoRA 实验。

## 实验计划 2026-06-23

当前基准状态：

```text
BiRNA-BERT + NUC tokenization + mean pooling + center pooling + MLP classifier
```

当前版本只做 NUC baseline，不包含：

```text
LoRA
BPE
FiLM
MoE
双视角融合
手工特征
```

实验策略：

1. 不跑合并数据集作为主要实验，即不把 11 个小分类混在一起作为默认主结果。
2. 所有实验优先按 11 个物种/组织小分类单独跑。
3. 每次模型结构更新后，先只跑人类 3 个数据集验证模型性能，并和 MKE-ResNet 对应数据集结果比较：

```text
Human_Brain
Human_Kidney
Human_Liver
```

4. 人类 3 个数据集确认有提升或至少行为合理后，再扩展到全部 11 个小分类：

```text
Human_Brain
Human_Kidney
Human_Liver
Mouse_brain
Mouse_heart
Mouse_kidney
Mouse_liver
Mouse_test
rat_brain
rat_kidney
rat_liver
```

5. 每个小分类都使用自己的 `train.csv` 做 5 折交叉验证，并保留自己的 `test.csv` 作为独立测试集。
6. 每折按验证集最优模型保存，再在独立测试集上评估；最终汇总 5 折 independent test 指标的 mean/std。

后续模型开发顺序：

```text
1. NUC frozen baseline
2. NUC full fine-tune
3. NUC LoRA
4. BPE baseline
5. NUC + BPE concat
6. NUC + BPE + FiLM
7. NUC + BPE + FiLM + MoE
```

每一步只增加一个核心变量，避免多个改动同时加入后无法判断性能来源。

当前决定：暂不做 full fine-tune，下一步优先做 `NUC + LoRA`。

## 项目结构

```text
BiRNA_m6A/
├── pretrained/
│   └── birna-bert-model/
├── data/
│   └── m6A_41bp/
├── src/
│   ├── train_birna_nuc_baseline.py
│   ├── dataset_utils.py
│   ├── model_birna_nuc.py
│   └── metrics_utils.py
├── outputs/
├── logs/
├── requirements_birna.txt
└── README_run.md
```

说明：你给出的数据来源 `m6A_dataset/data/birna-bert-model/` 在当前项目里未找到。本次整理使用当前项目真实存在的 41nt m6A 数据：

```text
m6A_dataset/data/mke_resnet_41nt_sequence_label/
```

并复制到：

```text
BiRNA_m6A/data/m6A_41bp/
```

## 安装依赖

云服务器建议优先按环境说明安装：

```bash
cd BiRNA_m6A
bash env/create_conda_env_cuda121.sh
conda activate birna_m6a
```

更多说明见：

```text
env/README_env.md
```

如果已经有可用的 PyTorch 环境，也可以只安装普通依赖：

```bash
cd BiRNA_m6A
pip install -r requirements_birna.txt
```

注意：`requirements_birna.txt` 不包含 `torch`，因为云服务器需要按 CUDA 版本安装对应 PyTorch wheel。
如果之前已经创建过 `birna_m6a` 环境，请补装 LoRA 依赖：

```bash
conda activate birna_m6a
cd BiRNA_m6A
python -m pip install peft==0.11.1
```

不要在训练脚本里联网下载模型或数据；脚本中的 Hugging Face 加载均使用 `local_files_only=True`。

## 合并数据集 baseline 命令，不作为当前主实验

```bash
cd BiRNA_m6A

python src/train_birna_nuc_baseline.py \
  --model_dir ./pretrained/birna-bert-model \
  --tokenizer_dir ./pretrained/birna-bert-model \
  --data_dir ./data/m6A_41bp \
  --output_dir ./outputs/birna_nuc_baseline \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4 \
  --seed 42 \
  --freeze_backbone
```

`--freeze_backbone` 传入时冻结 BiRNA-BERT backbone，只训练 MLP 分类头；不传时会微调整个 backbone。

当前主实验不建议优先跑上面的合并数据集命令。上面命令会读取 `data/m6A_41bp/` 根目录下的 `all_train.csv/all_test.csv`，用于快速 sanity check 可以，但不作为与 MKE-ResNet 小分类结果对比的主结果。

## Human_Brain 五折交叉验证

默认使用：

```text
data/m6A_41bp/Human_Brain/train.csv
```

做分层 5 折交叉验证，并保留：

```text
data/m6A_41bp/Human_Brain/test.csv
```

作为独立测试集。每一折都会临时保存该折验证集最优模型，加载该模型在同一个独立测试集上评估一次，然后默认删除 `best_model.pt` 以节省磁盘空间，最后输出 5 折 test 指标均值和标准差。

```bash
cd BiRNA_m6A

python src/train_birna_nuc_cv.py \
  --model_dir ./pretrained/birna-bert-model \
  --tokenizer_dir ./pretrained/birna-bert-model \
  --data_dir ./data/m6A_41bp/Human_Brain \
  --output_dir ./outputs/human_brain_5fold_birna_nuc \
  --folds 5 \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4 \
  --seed 42 \
  --freeze_backbone
```

输出结构：

```text
outputs/human_brain_5fold_birna_nuc/
├── fold_1/
│   ├── metrics.json
│   ├── train_log.csv
│   └── test_predictions.csv
├── ...
├── fold_5/
├── cv_summary.csv
└── cv_summary.json
```

说明：`best_model.pt` 训练过程中会临时生成，用于加载验证集最优模型做 test 评估；评估结果写出后默认删除。若确实需要保留权重，在命令末尾加：

```bash
--keep_best_model
```

## 人类 3 个数据集优先验证

Human_Brain：

```bash
python src/train_birna_nuc_cv.py \
  --model_dir ./pretrained/birna-bert-model \
  --tokenizer_dir ./pretrained/birna-bert-model \
  --data_dir ./data/m6A_41bp/Human_Brain \
  --output_dir ./outputs/Human_Brain_5fold_birna_nuc \
  --folds 5 \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4 \
  --seed 42 \
  --freeze_backbone
```

Human_Kidney：

```bash
python src/train_birna_nuc_cv.py \
  --model_dir ./pretrained/birna-bert-model \
  --tokenizer_dir ./pretrained/birna-bert-model \
  --data_dir ./data/m6A_41bp/Human_Kidney \
  --output_dir ./outputs/Human_Kidney_5fold_birna_nuc \
  --folds 5 \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4 \
  --seed 42 \
  --freeze_backbone
```

Human_Liver：

```bash
python src/train_birna_nuc_cv.py \
  --model_dir ./pretrained/birna-bert-model \
  --tokenizer_dir ./pretrained/birna-bert-model \
  --data_dir ./data/m6A_41bp/Human_Liver \
  --output_dir ./outputs/Human_Liver_5fold_birna_nuc \
  --folds 5 \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4 \
  --seed 42 \
  --freeze_backbone
```

与 MKE-ResNet 比较时，优先对齐每个小分类的 independent test 指标：

```text
ACC
MCC
AUC
AUPRC
F1
Precision
Recall
```

## NUC + LoRA 五折交叉验证

LoRA 版本保持 NUC 输入、mean pooling、center pooling、MLP classifier 不变，只在 BiRNA-BERT backbone 的 attention `Wqkv` 模块上加入 LoRA 适配器。

默认 LoRA 参数：

```text
r = 8
lora_alpha = 32
lora_dropout = 0.05
target_modules = Wqkv
```

Human_Brain LoRA：

```bash
python src/train_birna_nuc_cv.py \
  --model_dir ./pretrained/birna-bert-model \
  --tokenizer_dir ./pretrained/birna-bert-model \
  --data_dir ./data/m6A_41bp/Human_Brain \
  --output_dir ./outputs/Human_Brain_5fold_birna_nuc_lora \
  --folds 5 \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4 \
  --seed 42 \
  --freeze_backbone \
  --use_lora \
  --lora_r 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules Wqkv
```

Human_Kidney LoRA：

```bash
python src/train_birna_nuc_cv.py \
  --model_dir ./pretrained/birna-bert-model \
  --tokenizer_dir ./pretrained/birna-bert-model \
  --data_dir ./data/m6A_41bp/Human_Kidney \
  --output_dir ./outputs/Human_Kidney_5fold_birna_nuc_lora \
  --folds 5 \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4 \
  --seed 42 \
  --freeze_backbone \
  --use_lora \
  --lora_r 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules Wqkv
```

Human_Liver LoRA：

```bash
python src/train_birna_nuc_cv.py \
  --model_dir ./pretrained/birna-bert-model \
  --tokenizer_dir ./pretrained/birna-bert-model \
  --data_dir ./data/m6A_41bp/Human_Liver \
  --output_dir ./outputs/Human_Liver_5fold_birna_nuc_lora \
  --folds 5 \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4 \
  --seed 42 \
  --freeze_backbone \
  --use_lora \
  --lora_r 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules Wqkv
```

说明：`--use_lora` 开启后，PEFT 会只开放 LoRA adapter 参数；分类头仍正常训练。这里保留 `--freeze_backbone` 是为了实验语义清楚，即不做 full fine-tune。

如果 tokenizer 不在 `./pretrained/birna-bert-model`，请指定真实 tokenizer 路径，例如：

```bash
--tokenizer_dir ./pretrained/birna-tokenizer
```

## 数据读取规则

脚本会自动扫描 `data/m6A_41bp/`，支持：

- `csv/tsv` 且包含 `sequence,label`；
- 每行一个序列，标签从文件名 `pos/positive/neg/negative` 推断；
- fasta；
- 文件名或目录名中包含 `train/test/val/dev` 的原始划分。

如果发现 `all_train.csv` 和 `all_test.csv`，会优先使用这对总表，避免递归读到各组织子目录造成重复样本。若原始划分没有验证集，会从训练集按标签分层切出 10% 作为验证集；若没有任何划分，则按 8:1:1 自动分为 train/val/test。

序列清洗规则：

- 转大写；
- `U -> T`；
- 去掉空白；
- 仅接受 A/C/G/T；
- 长度必须为 41；
- 长度不符或含非法字符的样本跳过，并在训练启动日志中输出统计。

## 输出文件

训练输出保存在 `--output_dir`：

```text
metrics.json
train_log.csv
test_predictions.csv
```

默认不长期保留 `best_model.pt`：脚本会在评估完成、`metrics.json` 和 `test_predictions.csv` 写出后删除最优权重文件，避免大量实验占用磁盘空间。需要保留权重时使用 `--keep_best_model`。

`test_predictions.csv` 包含：

```text
sequence,label,prob,pred
```
