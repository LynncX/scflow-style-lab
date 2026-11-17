# 服务器环境说明

基线推荐使用 Python 3.10 + CUDA 12.1。SCFlow 官方在 Ubuntu 22.04.5 + CUDA 12.1 上测试。

## 安装

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

验证：

```bash
python scripts/check_gpu.py
```

mock 流程不需要 GPU，也不需要 `requirements.txt` 中的大模型依赖：

```bash
PYTHONPATH=src bash scripts/check.sh
```

## 资源下载

先下载权重，再按需下载数据：

```bash
python scripts/download_assets.py --group ckpt
python scripts/download_assets.py --group data
python scripts/download_assets.py --group all
```

脚本支持断点续传与 SHA256 校验。下载后可用以下命令检查：

```bash
python scripts/download_assets.py --verify-only --group all
```

默认输出到 `ckpts/` 和 `dataset/`。这两个目录已在 `.gitignore` 中排除。

## GPU 上跑真实基线

1. 下载权重：`python scripts/download_assets.py --group ckpt`
2. 用样例图先 smoke test：

```bash
PYTHONPATH=src python src/runner.py configs/baseline_scflow.yaml
```

3. 批量实验时，按服务器实际路径新建 manifest，每行一个 JSON：

```json
{"sample_id":"001","content_path":"/path/to/content.png","style_path":"/path/to/style.png"}
```

真实模式需要的两个权重：

- `ckpts/scflow_last.ckpt`
- `ckpts/sd21-unclip-l.ckpt`

如果服务器上的路径不同，不要改 YAML，用环境变量或命令行覆盖：

```bash
export SCFLOW_CKPT=/data/ckpts/scflow_last.ckpt
export SCFLOW_UNCLIP_CKPT=/data/ckpts/sd21-unclip-l.ckpt
export SCFLOW_MANIFEST=/data/manifests/batch.jsonl
export SCFLOW_OUT_ROOT=/data/runs
```

也可以用命令行：

```bash
PYTHONPATH=src python src/runner.py configs/baseline_scflow.yaml \
  --set scflow.ckpt=/data/ckpts/scflow_last.ckpt \
  --set scflow.unclip_ckpt=/data/ckpts/sd21-unclip-l.ckpt
```

断点续跑使用固定输出目录：

```bash
PYTHONPATH=src python src/runner.py configs/baseline_scflow.yaml \
  --out-dir /data/runs/baseline_1step --resume
```

单个样本失败会写入 `errors.jsonl`，并继续处理后续样本。所有 stdout 同时保存到 `runs/.../log.txt`。

## 输出

每次运行在 `runs/<run_name>_<timestamp>/` 下生成：

- `predictions/<sample_id>/clip_pred_*.json`：CLIP 特征
- `predictions/<sample_id>/<sample_id>.png`：生成图
- `run.json`：配置、样本路径、NFE、耗时、显存等记录
