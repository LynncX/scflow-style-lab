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
- AdaIN、StyTr² 与 StyleID 已有薄适配代码及 `*.example.json`，但尚未准备环境、权重或做真实推理测试。配置默认指向微型 fixture，只用于入口 smoke，绝非正式实验清单。三者必须通过 `worker_command` 在独立进程执行，避免上游绝对模块名相互污染。
- 完整上游版本固定在 `manifests/upstreams.json`。共享 clone 不修改；源码、权重路径可在配置中替换。真实适配器在执行前校验源码版本／补丁并散列权重，续跑时权重变化也会拒绝复用。
- `patches/stytr2-inference.patch` 删除上游 import 时对 GPU 环境的覆盖；在 eval 模式下生成 `Ics` 后直接返回，不运行训练损失和两条 identity 重建；替换 `torch._six` 导入并修正 torchvision 版本比较。未改动生成 `Ics` 的计算路径，但仍需真实数值等价验证。补丁文件保留上游行尾及上下文空白以匹配源码。
- AdaIN／StyTr² 采用 RGB → 短边缩放 → 中心裁剪 → Tensor，默认 512；AdaIN alpha=1，无颜色保持，权重采用安全 state-dict 加载。SCFlow 原预处理未改，**尚不能据此宣称四方法预处理已统一或公平对比已完成**。
- 逐样本 `total_ms` 包含 worker 通信与输出落盘，首样本还包含模型加载；`model_load_ms` 单独记录。它不包含样本循环前的清单／资源指纹校验，不是整条 CLI 的墙钟耗时。后续样本为热模型执行，不能混为纯网络耗时。显存是 worker 的分配峰值，不是整机总显存。
- StyleID 接入的是原版 `run_styleid.py`，未替换成 diffusers 实现；补丁支持显式单配对、sample ID 命名、可配置 seed 和复用模型。示例保留上游 seed=22、50 步反演／采样、start_step=49；不接受未实现的步数配置。每对重新初始化随机种子，区别于上游整个目录任务只初始化一次，不能默认视为同一结果口径。
- StyleID 首版**禁用磁盘反演缓存**，每对重新进行内容和风格反演，全部计入 `total_ms`，不存在隐藏缓存命中；仅模型常驻复用。适配器在反演阶段也关闭 autograd，模型训练逻辑不在本工程范围。
- StyleID 仅支持 CUDA，当前没有 CPU 回退／显存卸载策略。NFE 用 UNet forward hook 统计，包含反演，不等同于 SCFlow 的 Flow NFE；不得直接横比。
- StyleID 需另备本地 CLIP 文本编码器目录（模型权重、config、vocab、merges 等，见 `text_encoder`），明确禁用 CLIP 隐式下载。该目录全部文件与 SD checkpoint／模型配置纳入运行指纹。安全 checkpoint 加载使用 `weights_only=True`，需支持该参数的 torch；原版老环境未必兼容，尚未给出经过真实验证的环境版本组合。
- 控制层及当前 AdaIN／StyTr² worker 的哈希实现使用 Python 3.11+；原版 StyleID 的 Python 3.8 环境不能直接加载本工程的 Python 3.10+ 类型语法。环境准备不能原样照抄上游旧 YAML；目前未授权安装，也未验证版本组合。
- StyleID 仍保留上游“中心裁方形后 LANCZOS 缩放”的输入处理，与 AdaIN／StyTr² 的缩放再裁剪不完全相同，尚不能作公平主表。生成流程与原版的数值等价性仍需真实 GPU 验证。

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

preflight 检查清单图片、固定源码／补丁、非空权重、解释器和模块可发现性，并报告磁盘空间；不实际 import torch，也不验证 CUDA、权重兼容性或真实模型显存。现阶段支持 mock／AdaIN／StyTr²／StyleID 的预检，SCFlow 仍沿用已有检查入口。

## 统一评测输入检查（尚未接研究指标）

```bash
python scripts/evaluate.py runs/adain_smoke
# mock 必须明确授权，导出仍标记 simulated，不计算 ArtFID 等指标。
python scripts/evaluate.py runs/mock_adain --allow-simulated
```

- 导出 `evaluation/inputs.json`：按运行配置快照中的原始配对顺序记录内容／风格／预测路径、预测 SHA256、方法、清单标识及分母。校验快照指纹、原图哈希、方法标记、PNG 容器头尾和尺寸；不冒充完整像素解码检查。
- 默认拒绝失败或缺失样本；`--allow-failures` 仅用于明确导出排除清单的非完整子集，不能将不同子集直接拼成公平主表。已有成功记录但预测丢失／格式错误时始终报错，不静默排除。
- mock 输出默认拒绝，需 `--allow-simulated`。此入口**不生成 `metrics.json`**；真实指标实现、参考集合与失败口径仍等待服务器原 800 组评价材料，不重新定义同名 ArtFID。
- 检查目前要求运行时的原图仍可访问；不是可迁移的离线结果包。PNG 内容标识用于后续评测追溯，不构成结果真实性证明。

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
