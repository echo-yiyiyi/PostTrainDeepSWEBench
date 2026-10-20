# 实验结果查看器

只读的本地 Web 服务，Python 3.10+ 标准库即可运行，无需 pip/npm 安装。界面参考
[archipelago/view-app](https://github.com/echo-yiyiyi/archipelago/tree/main/view-app)
的可折叠轨迹与工具输入/输出展示方式，按本仓库的结果结构重新实现，不含 ASR、ER。

```bash
# 在仓库根目录运行，然后打开 http://127.0.0.1:8765
python view-app/app.py

# 也可 cd view-app 后直接 python app.py
# 指定结果根目录、某个实验组，或者单次实验目录
python view-app/app.py --results-dir /path/to/results
python view-app/app.py --run-dir /path/to/results/deepswe_smoke/53928242 --port 8766
```

默认从仓库 `.env` 读取 `POST_TRAIN_BENCH_RESULTS_DIR`，其次读取环境变量，最后使用仓库
`results/`（支持它是符号链接）。不会执行/source `.env`。目录扫描默认最多四层，找到实验
目录后停止向内递归；更深的结果目录可用 `--scan-depth 6`。点击“刷新”重新扫描并读取结果。

在远程服务器启动时，可以在自己电脑上转发端口：

```bash
ssh -L 8765:127.0.0.1:8765 user@server
```

随后打开本机的 `http://127.0.0.1:8765`。默认只监听服务器回环地址；`--host` 可指定监听地址。
该服务没有身份认证，结果和原始日志可能包含敏感数据，因此建议使用默认监听和 SSH 转发。

## 页面

- **概览**：状态、执行时间、已保存指标、run_config、prompt、CLI 版本、各 judge 原始判决和多轮判决。
  指标优先显示根目录的 `metrics_averaged_exact_match.json`、`metrics_averaged.json`，然后是
  `metrics.json`；这里只展示文件，不复现 `scripts/collect.py` 的完整计分与 judge 回退逻辑。
- **Agent 轨迹**：默认优先读取 `solve_out_sanitized.txt`，否则读取 `solve_out.txt`；可以切换
  parsed 文件和 judge 轨迹。支持 Codex item/旧版事件、Claude、Gemini stream、OpenCode、Cursor
  工具事件，以及 `messages` 格式的 JSON trajectory。未知事件保留原始 JSON，普通文本/损坏行
  保留为日志卡片。工具输入、输出、状态、原始事件均可展开与复制；同一调用的开始/结束按 ID
  合并，Codex 新 thread 的 ID 不会与旧 thread 混合。按类型或全文搜索，每页 100 条。
- **评测对话**：单独选择最终评测或训练期间试评测，按轮次展开模型输入上下文、输出及工具返回。工具调用与返回按 ID 合并为和 Agent 轨迹
  一致的可折叠卡片，Bash 命令直接高亮显示。顶部可统一切换“格式化视图 / 原始 JSON”；参数
  JSON 无效时标记格式异常并保留原文，原始视图保留完整消息字段。
  DeepSWE 优先读取 `mini-swe-agent.trajectory.json`；输入是按历史消息重建，非原始 HTTP 包。
  Inspect JSON 日志存在 model 事件时直接展示其保存的输入/输出；没有记录时明确提示缺失。
- **最终工作空间**：默认直接展示容器路径 `/home/ben/task/` 下最终保留的目录与文件，点击即可
  查看代码、配置和文本。DeepSWE 映射自 `training_home/task/`，标准实验映射自 `task/`；未保存
  工作目录时明确提示缺失。可切换到“实验记录文件”查看轨迹、评估日志等。文本每段 256 KiB，
  支持翻页和复制；行号是当前段内行号。二进制仅显示信息。
- **语法高亮**：代码文件按扩展名识别 Python、Bash、JavaScript、JSON、YAML、HTML/CSS、Diff；
  轨迹中的 command/cmd 字段和 shell 调用按 Bash 高亮，常见 Python heredoc 中的代码按 Python
  高亮。代码块带语言标签、行号和水平滚动，复制保留原文。
  JSON/JSONL/TXT 可以“作为轨迹查看”，用于嵌套 DeepSWE 评估轨迹。

## 实际保存了什么

标准 `src/run_task.sh` 会先单独保存最终模型到 `final_model/`，清理 task 中 Hugging Face
模型目录，然后复制剩余工作目录到 `task/`。因此它是**清理后的工作目录快照**，不是整个容器
文件系统，也不能恢复执行中已经删除的临时文件。

DeepSWE 保留 `training_home/task/`，保存产物时创建 `task` 链接，完成时创建 `final_model`
链接。查看器允许实验目录内的符号链接，拒绝指向当前实验目录以外的链接。失败或尚未完成的实验
也可以查看已经写出的文件；没有轨迹/指标时明确显示缺失，不推断成功或失败。

本服务不修改实验文件，不执行轨迹中的命令，不执行浏览到的 HTML/脚本。轨迹解析结果在内存中
缓存最近两个文件，并根据文件大小/修改时间失效；首次读取会解析完整轨迹，超大型日志首次打开
可能较慢。所有原始事件仍然保留。文件预览按字节分段，跨段的 UTF-8 字符可能显示替换字符。

## 验证

```bash
python -m unittest discover -s view-app -p 'test_*.py' -v
node --check view-app/static/app.js  # 可选；运行服务不依赖 Node
```

语法高亮使用本地随附的 [Prism 1.29.0](https://prismjs.com/)，无需浏览器连接 CDN。
源文件位于 `static/vendor/`，许可证见 `static/vendor/PRISM-LICENSE`。
