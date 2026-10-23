# JustFit 2.3

面向 Apple silicon 的本地推理项目，协调 CPU、GPU 与 ANE，支持并发解码、
持久化前缀缓存及按条件启用的 ANE 图像编码。基于 Splash 1.3.0，保留上游致谢和许可证。

当前为未公开发布的候选。验收状态以与产物完整哈希匹配的外部回执为准；
安装软件包不会自动注册或启动服务。论文和网站仍保留占位。

从 [Quickstart](QUICKSTART.md) 开始。Python wheel/sdist 提供 `justfit-llm`
命令、源码与依赖；原生程序需要在支持的 Apple silicon 环境构建。
安装 wheel 后运行 `justfit serve`，可创建工作区、查找工具链并编译、下载固定版本权重，然后在 `127.0.0.1:8000` 启动。需先具备兼容的 Xcode/CoreAI SDK 和 Metal 工具链；不会自动安装开发工具。ANE 编译图和本机缓存需要另行准备。

也可显式提供与此源码包匹配的预编译运行时及SHA256，跳过本机编译；参见[运行时分发](docs/RUNTIME_DISTRIBUTION.md)。当前未配置公开默认下载地址；不能把本候选视作已上线的免工具链安装服务。

`justfit serve` 可为至少24 GiB的 M4/M5/M6 自动选择预设；`scripts/run_local.py` 仍要求显式选择。参考配置：M4 Pro 使用 CPU 辅助 prefill；M5 Pro
开启 GPU A8 prefill/verify；M6 额外开启语言 ANE FFN prefill。
M6 有文本 prefill 时图像使用 GPU；无 prefill 的 decode 阶段才允许视觉 ANE。
M4 Pro/M5 Pro 可在文本 prefill 或 decode 时按条件使用视觉 ANE。

4K/8K 是图像 patch 容量，不是文本上下文限制。缺少合格本机 profile、
资产或内存准入时使用 GPU。目标是释放 GPU 给文本工作，不保证每张图片更快。
按照 [ANE 资产准备指南](docs/ANE_PROVISIONING.md) 导出、离线编译并验收本机资产。

[Thinking budget](docs/THINKING_BUDGET.md) 由原生运行时执行硬限制；
前端与原生程序必须成对升级。外部 broker 如会重写请求，需要单独检查
历史保留、工具消息配对和预算转发，参见[可选适配器](service/broker-integration/README.md)。

参见[Python 安装](docs/PYPI.md)、[管理服务与回滚](docs/DEPLOYMENT.md)、
[验证与限制](docs/VALIDATION.md)、[许可证](LICENSE)及[第三方声明](NOTICE)。

## 引用

如果 JustFit 对您的研究有帮助，请引用我们的论文：

Yuhua Chen. *JustFit v2: Long-Context Local LLM Serving with Just-in-Time State Management*. 未发表稿，2026。

```bibtex
@unpublished{chen2026justfitv2,
  author = {Chen, Yuhua},
  title = {{JustFit v2}: Long-Context Local LLM Serving with Just-in-Time State Management},
  year = {2026},
  note = {Unpublished manuscript}
}
```

[CITATION.cff](CITATION.cff) 提供机器可读的引用信息。论文公开后补充链接及编号。

预编译 ANE 4K／8K 资产可独立下载并校验安装，具体芯片和系统版本必须匹配；
流程见 [ANE 下载与安装](docs/ANE_DOWNLOADS.md)。安装器不编译图，也不清理已有缓存。
