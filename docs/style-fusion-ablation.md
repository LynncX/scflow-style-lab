# 多参考风格融合消融

使用原 COCO × WikiArt 清单。A 组取 `style_path`，它也是 `style_paths` 的第一项。B/C 两组按原顺序读取三张同风格参考。三组固定 Euler 1 步、种子 `2025` 和 `512×512` 输出。

| 组别 | 风格表示 | 配置 |
| --- | --- | --- |
| A | 第一张参考的 CLIP 特征 | `configs/baseline_scflow.yaml` |
| B | 三个原始 CLIP 特征的算术均值 | `configs/experiments/scflow_three_mean.yaml` |
| C | 按与三参考均值的余弦相似度做 softmax，加权求和 | `configs/experiments/scflow_three_weighted.yaml` |

C 组固定 softmax 温度 `0.1`。PPT 给出了按风格中心相似度计算归一化权重的思路，本实验将温度作为预先固定的配置。以原始 CLIP 特征计算风格中心并求加权和，余弦归一化只用于计算相似度。

## 107 上的试跑

在独立代码目录中使用原 SCFlow 虚拟环境与权重。以下命令由 GPU 值班人在同一张卡上顺序执行。

```bash
base=/home/wang/XuanyiLi/scflow-style-lab
data=/data/wang/XuanyiLi/Data/benchmark/coco_wikiart_v1
out=/data/wang/XuanyiLi/Results/scflow_member2_pilot
export XDG_CACHE_HOME="$base/.cache"
export HF_HOME="$base/.cache/huggingface"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src

for variant in mean weighted; do
  "$base/.venv/bin/python" src/runner.py "configs/experiments/scflow_three_${variant}.yaml" \
    --set "manifest=$data/pilot.jsonl" \
    --set "scflow.ckpt=$base/ckpts/scflow_last.ckpt" \
    --set "scflow.unclip_ckpt=$base/ckpts/sd21-unclip-l.ckpt" \
    --out-dir "$out/$variant" --resume || exit $?
  "$base/.venv/bin/python" scripts/verify_style_fusion.py "$out/$variant" --expected-count 8 || exit $?
done
```

验收时检查两组的 8 个样本 ID、参考顺序和 NFE 是否一致。每组须生成 8 张可解码 RGB PNG。权重写入 `records.jsonl`，并由验收脚本导出 `evaluation/style_weights.csv`。B 组各参考权重为 `1/3`，C 组每个样本的三项权重之和为 `1`。

全量跑使用同样配置，将清单换成原 `manifest.jsonl`、验收数量改为 `800`，并使用独立输出目录。A 组可复用已有单参考固定 1 步的图像和质量指标。同硬件与缓存口径的时延比较需要另行采集。

## 完整方法接入

`src/backends/style_fusion.py` 提供有序参考选择与三参考融合。SCFlow 后端将融合结果作为单个 `[1, 768]` 风格向量与内容向量拼接。加权配置可复用于自适应积分实验，阈值按该实验的验证集确定。
