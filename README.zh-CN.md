<div align="center">
  <picture>
    <source media="(max-width: 600px)" srcset="docs/assets/justfit-banner-mobile.svg">
    <img src="docs/assets/justfit-banner.svg" alt="JustFit — 面向 Apple silicon 的长上下文本地 LLM 推理" width="100%">
  </picture>

  [![版本](https://img.shields.io/badge/version-2.3.1-087f78)](CHANGELOG.md)
  [![许可证：Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-246a96)](LICENSE)

  **[开始使用](#开始使用) · [实测结果](#实测结果) · [文档导航](#文档导航) · [English](README.md)**
</div>

JustFit 面向内存受限的 Mac，协调压缩上下文、常驻推测 draft、并发请求及 CPU/GPU/ANE，运行 **27B 语言与视觉模型**。所有计算共享同一份统一内存预算。

**2.3.1 已发布到 [GitHub Releases](https://github.com/YuhuaBillChen/justfit-llm/releases/tag/v2.3.1)**，包括源码、Python 包和 Apple 公证的原生运行时；[2.3.0](https://github.com/YuhuaBillChen/justfit-llm/releases/tag/v2.3.0) 的六份 ANE 包原样适用。[正式 PyPI 2.3.1](https://pypi.org/project/justfit-llm/2.3.1/) 也已上线，按下方命令安装。

## JustFit 做什么

- **保留长历史，也保留 draft。** warm-prefix 实验在两条同时存活的请求中达到 320K 聚合页容量，v2 保持推测 draft 常驻。这是 B2 已测容量，不是单请求 320K 或冷启动 prefill 的保证。
- **让并发有空间。** 限制临时工作区、共享压缩页，在调度、持久化和冷重启间保留已提交的复合状态。
- **让空闲加速器接活。** M4 Pro 使用 CPU 辅助 prefill，M5 Pro/M6 使用 GPU A8，并在阶段和内存允许时使用 ANE。视觉卸载旨在把 GPU 留给文本，不保证每张图更快。
- **提供熟悉的接口。** OpenAI 兼容聊天、图像、流式输出、结构化输出，以及原生执行的 thinking budget 硬限制。

## 实测结果

<a href="docs/assets/system-comparison.png"><img src="docs/assets/system-comparison.png" alt="M4 Pro、M5 Pro、M6 的四代系统对比：已测上下文容量、16K prefill 和 decode 速度" width="100%"></a>

**容量和速度来自独立实验。** 实心柱是已测 B1 边界，斜线延伸到单独测量的 B2 总量。v1/v2 按各自 warm-prefix 协议均达到 320K 页容量；v2 的 draft 常驻。速度采用八道固定 SPEED-Bench 16K 题，每题强制输出 512 token；权重格式、decoder 和设备配置不同。M4 Pro/M5 Pro 是物理 24 GiB；M6 是物理 32 GiB，仅 v1/v2 的 B2 容量测试锁住 8 GiB，速度测试使用物理 32 GiB。

在这些配置的 16K 实验中，v2 的 decode 中位速度为 v1 的 **2.0–2.5 倍**；相对上游 Splash 1.3.0，prefill 提升 **9–46%**，单路 decode 下降 **2–9%**。这是完整系统对比，不能当作单机制收益或芯片代际排名。

[完整协议、速度和限制 →](docs/RESULTS.md)

## 开始使用

### 环境

- **Apple silicon、macOS 27.0+、至少 24 GiB。** 参考配置为 M4 Pro、M5 Pro、M6，其他芯片变体没有逐一验收。
- 推荐 **Python 3.12**；包声明兼容 Python 3.12–3.14。
- 下方预编译安装路径无需本机编译。源码构建另需兼容的 **Xcode/CoreAI SDK 与 Metal 编译工具链**。
- 固定模型包约 **17.4 GB**，还需为依赖、编译输出和缓存留空间；已检查的模型 revision 无需 Hugging Face token 即可访问。

Linux 可以安装 Python 工具，但不能运行这里的原生推理。独立 GPU 引擎的 macOS 26.4 源码目标不等于完整 Python/CoreAI 路径支持 26.4。

### 安装已测试版本

建立独立环境，从正式 PyPI 安装固定版本：

```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install justfit-llm==2.3.1
justfit verify
justfit doctor
justfit serve \
  --runtime-manifest https://github.com/YuhuaBillChen/justfit-llm/releases/download/v2.3.1/runtime-manifest.json \
  --runtime-sha256 <发布页 SHA256SUMS 中 runtime-manifest.json 的 SHA256> \
  --ane-manifest https://github.com/YuhuaBillChen/justfit-llm/releases/download/v2.3.0/ane-index.json \
  --ane-manifest-sha256 fd1b71cfa9d4e785741dc2e0ba4d43f2a4dd556a62c0a61f37838b7f9e3ec99d
```

首次运行按以上参数下载并校验配套的预编译运行时、匹配的 ANE 包和固定模型，然后在 **`http://127.0.0.1:8000`** 提供服务。ANE 包要求精确匹配芯片与 macOS build `26A434`，其他系统可省略 ANE 两个参数使用 GPU 视觉。后续运行 `justfit serve` 复用已安装内容；Ctrl+C 停止前台服务。

未提供运行时参数且没有已安装运行时的首次运行会走源码编译路径，需要开发工具。[Quickstart](QUICKSTART.md) 提供源码构建、端口和模型配置。

### 发一个请求

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "incoai/Qwen3.8-27B-Splash",
    "messages": [{"role": "user", "content": "17 乘以 19 是多少？"}],
    "max_tokens": 64,
    "reasoning_effort": "none"
  }'
```

模型 ID 保留上游运行时包的名称，程序与 CLI 品牌是 JustFit。默认权重 revision 为 `9d27070b71f7142c6b6025f03ac011d70a73cb48`。

## 四个机制

| 机制 | 作用 |
| --- | --- |
| **KVExec** | 以有界重建和旋转域注意力消费压缩历史。 |
| **PhaseSwap** | 管理组件和工作区生命周期，同时保持 draft 常驻。 |
| **StateTrans** | 在调度和持久化间保留已提交的 KV、循环状态和 draft 状态。 |
| **CrossFire** | 根据阶段所有权和内存准入协调 CPU、GPU 与 ANE。 |

### ANE 如何参与

| 参考预设 | 语言 prefill | 视觉 ANE 可用阶段 |
| --- | --- | --- |
| M4 Pro / `m4` | CPU 辅助 GPU | 文本 prefill 或 decode |
| M5 Pro / `m5` | GPU A8 | 文本 prefill 或 decode |
| M6 / `m6` | GPU A8＋ANE FFN | 无文本 prefill 的 decode 阶段 |

视觉 ANE 需要验收过的 4K/8K 资产、匹配的芯片/系统 build 和内存准入。**4K/8K 是图像 patch 数，不是文本 token 数或屏幕像素。** 空闲请求、不支持的输入或资源不足时用 GPU。M6 优先保证语言 ANE prefill。[ANE 下载与安装接口](docs/ANE_DOWNLOADS.md)。

## 文档导航

| 需求 | 文档 |
| --- | --- |
| 安装、构建、选端口和模型 | [Quickstart](QUICKSTART.md) · [Python 安装](docs/PYPI.md) |
| 看结果和验证范围 | [实验协议](docs/RESULTS.md) · [测试版验收摘要](docs/PREVIEW_VALIDATION.json) |
| 修改设备配置 | [配置](docs/CONFIGURATION.md) · [ANE 策略](ANE_POLICY.md) |
| 准备 ANE 与预编译运行时 | [ANE 安装](docs/ANE_DOWNLOADS.md) · [离线准备](docs/ANE_PROVISIONING.md) · [运行时分发](docs/RUNTIME_DISTRIBUTION.md) |
| 管理服务与接入 broker | [部署和回滚](docs/DEPLOYMENT.md) · [适配器](service/broker-integration/README.md) |
| 限制思考长度 | [Thinking budget](docs/THINKING_BUDGET.md) |
| 反馈和贡献 | [Issues](https://github.com/YuhuaBillChen/justfit-llm/issues) · [贡献指南](CONTRIBUTING.md) · [安全说明](SECURITY.md) |

默认仅监听本机。对外网络部署需要认证、TLS 和访问控制，见安全说明。`source/` 是原生运行时与前端，`vision-tools/` 是 ANE 工具，`service/` 是可选管理服务。

## 论文与引用

**JustFit v2: Long-Context Local LLM Serving with Just-in-Time State Management**，Yuhua Chen，2026。**[阅读预印本 PDF](https://github.com/YuhuaBillChen/justfit-llm/blob/main/paper/JustFit-v2.pdf)** · [论文与引用文件](paper/README.md)。此固定入口始终指向当前论文；arXiv 公开后再补链接。

**完整 AIME 2026：** M5 Pro 在相同 4-bit 权重与 TQ4 KV 下，A8 关闭答对 **86/90（95.6%）**，开启答对 **85/90（94.4%）**，覆盖全部 30 题、三个 seed。每臂各有 3 次输出达到上限，计入分母；该比较检验量化配置内的 A8 开关。

如果 JustFit 对你的研究有帮助，请引用：

```bibtex
@unpublished{chen2026justfitv2,
  author = {Chen, Yuhua},
  title = {{JustFit v2}: Long-Context Local LLM Serving with Just-in-Time State Management},
  year = {2026},
  month = oct,
  note = {Preprint},
  url = {https://github.com/YuhuaBillChen/justfit-llm/blob/main/paper/JustFit-v2.pdf}
}
```

[机器可读引用信息](CITATION.cff) · [JustFit v1 论文](https://arxiv.org/abs/2609.17475)

## 致谢与许可证

JustFit 原生运行时基于 **[Splash 1.3.0](https://github.com/incoai/splash)**。语言 ANE FFN、KV 分页分层和复合状态持久化框架来自上游；JustFit 扩展其集成、恢复点保留和资源协调。上游作者和实现贡献保留在 [NOTICE](NOTICE) 与[第三方声明](source/THIRD_PARTY_NOTICES)中。权重和衍生资产保留各自许可证。

代码采用 [Apache 2.0](LICENSE)。JustFit 的问题由本项目维护者处理，请向本仓库提交反馈。
