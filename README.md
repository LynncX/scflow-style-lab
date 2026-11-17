# SCFlow Style Lab

课程设计 baseline：单参考、固定 1 步 Flow Matching 风格迁移。本地用 mock 验流程，服务器用 SCFlow 权重跑 GPU。

## 本地检查

```bash
PYTHONPATH=src bash scripts/check.sh
```

## 服务器运行

详见 `ENVIRONMENT.md`。简要流程：

```bash
python scripts/download_assets.py --group ckpt
python scripts/check_gpu.py
python -m pip install -r requirements.txt
PYTHONPATH=src python src/runner.py configs/baseline_scflow.yaml \
  --out-dir runs/baseline_1step --resume
```

## 输出

每次运行在 `runs/<name>/` 下生成：

- `predictions/<id>/clip_pred_*.json`：CLIP 特征
- `predictions/<id>/<id>.png`：生成图片
- `run.json`：配置、耗时、NFE、显存
- `log.txt`：运行日志
- `errors.jsonl`：失败样本（如有）

权重和数据集不入库；代码已包含 SCFlow 推理所需源码。
