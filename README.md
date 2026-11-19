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

## 独立 worker 与真实算法接入（尚无 GPU 验证）

- 配置 `worker_command` 为现有算法环境的 Python argv 列表，如 `[".envs/adain/bin/python", "-u"]`；不经 shell 拼接。控制层启动一个常驻 worker，通过 JSONL 传递精确配对；逐样本异常不会销毁模型。worker 输出写入 `logs/worker.log`，运行结束／中断时回收子进程。
- 可在任意 mock 配置增加该字段，验证真实跨进程调度；不要求 torch。没有该字段时保留原进程内运行方式。
- AdaIN 与 StyTr² 已有薄适配代码及 `*.example.json`，但尚未准备环境、权重或做真实推理测试。配置默认指向微型 fixture，只用于入口 smoke，绝非正式实验清单。
- 完整上游版本固定在 `manifests/upstreams.json`。共享 clone 不修改；源码、权重路径可在配置中替换。真实适配器在执行前校验源码版本／补丁并散列权重，续跑时权重变化也会拒绝复用。
- `patches/stytr2-inference.patch` 删除上游 import 时对 GPU 环境的覆盖；在 eval 模式下生成 `Ics` 后直接返回，不运行训练损失和两条 identity 重建；替换 `torch._six` 导入并修正 torchvision 版本比较。未改动生成 `Ics` 的计算路径，但仍需真实数值等价验证。补丁文件保留上游行尾及上下文空白以匹配源码。
- AdaIN／StyTr² 采用 RGB → 短边缩放 → 中心裁剪 → Tensor，默认 512；AdaIN alpha=1，无颜色保持，权重采用安全 state-dict 加载。SCFlow 原预处理未改，**尚不能据此宣称四方法预处理已统一或公平对比已完成**。
- 逐样本 `total_ms` 包含 worker 通信与输出落盘，首样本还包含模型加载；`model_load_ms` 单独记录。后续样本为热模型执行，不能混为纯网络耗时。显存是 worker 的分配峰值，不是整机总显存。
- StyleID 当前只有 mock；真实接入需处理原版固定 seed、精确配对、反演缓存和常驻模型，未替换成 diffusers 实现。

```bash
# 只检查，不安装或下载。缺少资源时返回非零。
python scripts/bootstrap.py --check
python scripts/preflight.py configs/experiments/adain.example.json

# 显式准备独立源码副本，优先复用本机完整 clone（不改动来源）。
python scripts/bootstrap.py --prepare --local-root /path/to/shared/gitclone \
  --method adain --method stytr2
# 不传 --local-root 则从固定上游 URL clone；不会下载权重或安装环境。

# 用户另行准备好算法环境和权重之后，再执行真实 smoke。
PYTHONPATH=src python src/runner.py configs/experiments/adain.example.json \
  --out-dir runs/adain_smoke --resume
```

preflight 检查清单图片、固定源码／补丁、非空权重、解释器和模块可发现性，并报告磁盘空间；不实际 import torch，也不验证 CUDA、权重兼容性或真实模型显存。现阶段只实现 mock／AdaIN／StyTr² 的预检，SCFlow 仍沿用已有检查入口。

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
