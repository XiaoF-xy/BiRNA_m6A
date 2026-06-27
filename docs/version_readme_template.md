# Version Name

## 实验目的

说明该版本要验证的问题。

## 使用的数据集

说明数据集名称、物种/组织、窗口长度、train/test 划分。

## 使用的模型或方法

说明模型结构、输入特征、是否 LoRA、是否冻结 backbone。

## 主要修改点

说明相对上一版本具体改了什么。

## 训练命令

```bash
python train.py --version <version> --dataset <dataset> --seed 42
```

## 输出结果保存位置

```text
outputs/<version>/<dataset>/seed_<seed>/
```

## 与上一版本相比的变化

说明新增模块、参数变化和实验假设。

## 当前版本的实验结论或备注

记录当前结果、失败原因、待验证前提或下一步计划。

