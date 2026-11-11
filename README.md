# PeopleTrack

PeopleTrack 在 MOTS20 上跟踪行人，在 KITTI MOTS 上跟踪车辆和行人。输入为按帧编号的图像序列，输出为逐序列的 MOTS 掩码轨迹文件和提交 ZIP。项目包含最终推理权重、本地评估器及运行脚本。

## 安装

使用 Python 3.10。CUDA 配置使用 PyTorch 2.5.1、TorchVision 0.20.1 和 CUDA 12.1 版 PyTorch wheel，需要兼容的 NVIDIA 驱动。

Windows PowerShell：

```powershell
.\setup_env.cmd
.\run.cmd --check
```

Linux 或手动安装（在仓库根目录运行，使用 Python 3.10 创建环境）：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -I run.py --check
```

下面的 `python` 命令使用已激活的环境。Windows 也可使用 `.venv\Scripts\python.exe`，或将 `python -I run.py` 换成 `.\run.cmd`。

## 数据

请从 [MOTS20](https://motchallenge.net/data/MOTS/) 和 [KITTI MOTS](https://www.cvlibs.net/datasets/kitti/eval_mots.php) 获取图像与标注。默认目录如下，可用 `--data-root` 和 `--gt-root` 指定其他位置。

| 内容 | 默认路径示例 |
| --- | --- |
| MOTS20 训练图像 | `datasets/MOTSChallenge/train/MOTS20-02/img1/000001.jpg` |
| MOTS20 训练标注 | `datasets/MOTSChallenge/train/MOTS20-02/gt/gt.txt` |
| MOTS20 测试图像 | `datasets/MOTSChallenge/test/MOTS20-01/img1/000001.jpg` |
| KITTI 训练图像 | `datasets/KITTI/training/image_02/0014/000000.png` |
| KITTI 测试图像 | `datasets/KITTI/testing/image_02/0000/000000.png` |
| KITTI 训练标注 | `datasets/KITTI/instances_txt/0014.txt` |

配置文件中的相对数据路径以仓库根目录为基准。命令行传入的相对路径以终端当前目录为基准。若通过 `--data-root` 指定数据集位置，默认标注目录也随之调整。手动指定 `--gt-root` 时，MOTS20 指向包含 `MOTS20-02/` 等目录的文件夹，KITTI 指向直接包含 `0014.txt` 等文件的文件夹。

## 推理与评估

```bash
# 生成 MOTS20 和 KITTI MOTS 测试集提交包
python -I run.py --dataset mots20 --split test
python -I run.py --dataset kitti --split test

# 运行一条 KITTI 训练序列并计算本地指标
python -I run.py --dataset kitti --split train --sequences 0014 --evaluate --output results/kitti_0014

# 单独评估已完成的结果
python -I evaluate.py --run-dir results/kitti_0014

# 对两个数据集各运行 60 帧，验证推理、打包和评估
python -I smoke_test.py --frames 60
```

默认设备为 `cuda`；可用 `--device cpu` 运行推理或短序列检查。运行 `python -I run.py --help` 和 `python -I smoke_test.py --help` 查看路径覆盖及其他参数。KITTI 的 `--split validation` 选用配置对应的九条开发序列。

默认输出位置为 `results/<dataset>_<split>/`。其中 `submission/` 保存逐序列 TXT，`evaluation/METRICS.csv` 保存本地 HOTA、DetA、AssA、IDF1 等指标，`RUN_PROTOCOL.json` 记录运行配置。带 `SMOKE_PARTIAL` 的 ZIP 仅用于短序列验证。MOTS20 默认测试包包含 4 条序列；若需要包含训练和测试序列的 8 文件 ZIP，先分别完成全量 `train` 和 `test` 推理，再执行 `python -I package_mots20_8seq.py`。

这里的 `train` 指数据集划分。训练标注只用于可选评估，测试集成绩由相应评测平台计算。项目提供最终模型的运行流程，未打包历史训练和消融实验。

## 实现

YOLO26x-seg 和 BoT-SORT 生成初始轨迹。OSNet 提取外观特征，历史一致性规则拆分轨迹，ResNetLink 和候选排序连接片段，最后按类别细化掩码。两类连接均使用片段首末各两次可用观测。KITTI 车辆分支另用背景运动估计做几何补偿。

| 目录 | 内容 |
| --- | --- |
| `peopletrack_final/` | 推理流程、轨迹连接、几何补偿、掩码细化与评估 |
| `trackrefine/` | 外观特征、轨迹拆分、关联特征与排序 |
| `configs/`、`weights/` | 默认参数、检测器和各分支权重 |
| `vendor/`、`licenses/` | Ultralytics（AGPL-3.0）、Torchreid（MIT）、TrackEval（MIT）及许可文件 |

## 上传到 GitHub

`weights/` 中的 `.pt`、`.pth` 文件由仓库内的 `.gitattributes` 配置为 Git LFS。检测器权重超过 100 MiB，请在首次 `git add` 前安装 Git LFS，再从本目录推送到已建好的空仓库。参见 [GitHub 的 Git LFS 指南](https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage)。

```bash
git init -b main
git lfs install --local
git add .
git lfs ls-files
git commit -m "Add PeopleTrack inference and evaluation"
git remote add origin https://github.com/OWNER/REPO.git
git push -u origin main
```

将 `OWNER/REPO` 换成自己的仓库。克隆后如发现权重是 LFS 指针，执行 `git lfs pull` 下载实际文件，再运行环境检查。
