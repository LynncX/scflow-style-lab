# SCFlow Style Lab

课程设计 baseline：单参考、固定 1 步 Flow Matching 风格迁移。本地用 mock 验流程，服务器用 SCFlow 权重跑 GPU。

## 本地检查

```bash
PYTHONPATH=src bash scripts/check.sh
```

## 对比实验最小闭环（仅 mock）

```bash
bash scripts/test_mock.sh
PYTHONPATH=src python src/runner.py configs/experiments/mock_adain.json \
  --out-dir runs/mock_adain --resume
```

- `configs/experiments/mock_{scflow,adain,stytr2,styleid}.json` 共用同一配对清单，生成 512×512 模拟 PNG；不调用真实算法，不计算研究指标。
- mock 沿用真实 runner 的配对、落盘及续跑逻辑；非 SCFlow 方法不生成 CLIP 特征，NFE 为 `null`。所有样本记录和汇总标记 `simulated`。
- `config.json` 保存最终配置、解析后的配对、清单及输入文件 SHA256；`provenance.json` 记录初次启动时的主仓库版本和 Python 信息，尚不验证权重、上游或完整环境。
- `records.jsonl` 逐样本追加并同步落盘，是中断恢复依据；`run.json` 保留原汇总格式，展示各样本最近一次结果。失败样本、输出缺失或空文件的样本会重试。
- `--resume` 仅复用成功且产物存在的记录；配置、清单或输入内容变化将拒绝复用。旧目录没有 `config.json` 时请使用新输出目录，不自动迁移。已有目录未指定 `--resume` 时拒绝覆盖。
- `data_root` 默认仓库根目录，清单中的相对图片路径按它解析；`sample_id` 必须唯一且不能含路径分隔符。输入缺失在执行前报错。
- 有失败样本时 CLI 返回非零退出码。单个输出目录只允许一个进程写入；目前不支持并发写入及产物内容校验。源码变化尚不自动阻止续跑，修改执行逻辑后应使用新目录。
- 测试使用临时目录，无 GPU、无权重、无下载或环境安装；可用 `PY=/path/to/python` 指定现有环境。

本轮未改变 SCFlow 推理实现，真实 GPU 和服务器 800 组评价仍待后续验证。

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
