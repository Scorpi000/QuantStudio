# Repository Guidelines（仓库贡献指南）

## 项目结构与架构
QuantStudio 是基于 Python 3.12+ 的量化研究框架，采用 GPLv3 许可证。源码位于 `QuantStudio/`：`Core/` 提供计算图引擎；`Factor/`、`BackTest/`、`Risk/`、`PortfolioConstructor/` 分别负责因子、回测、风险与组合优化；`Tools/` 提供通用工具，`Resource/` 存放资源。测试位于 `tests/`，示例位于 `examples/`，MCP 服务位于 `mcp/`，开发脚本位于 `scripts/`。

量化计算通过 `Node` 组成 DAG，由引擎执行。修改节点时遵循 `init_compute → prepare_compute → compute` 生命周期；公共接口通过各模块的 `api.py` 暴露。

## 开发与验证命令
在仓库根目录的 Python 虚拟环境中运行：

```bash
python -m pip install -r requirements.txt          # 安装核心与文档依赖
python -m pip install -e .                        # 可编辑安装
python -m pip install -r requirements_optional.txt # 安装可选集成依赖
python -m unittest discover -s tests -p "test_*.py" # 全量测试
python -m unittest discover -s tests -p "test_Core_Engine.py" # 定向测试
python scripts/build_docs.py                      # 构建文档
python scripts/build_docs.py --serve              # 本地预览文档
```

## 编码与协作约定
使用 UTF-8 编码和四空格缩进，匹配相邻代码的命名风格；保留现有公共 API 与 `__QS_*__` 标识符。参数沿用 Pydantic 模型，日志使用 `self.Logger` 或 `__QS_Logger__`。仓库未配置统一格式化或静态检查工具。

遵循 `CLAUDE.md` 的原则：实现前说明关键假设与权衡；遇到影响实现的歧义先澄清；只实现必要功能，只修改与任务相关的代码，只清理本次改动引入的冗余。多步骤任务先明确计划及验证标准，并持续验证至目标达成。

## 测试规范
使用 `unittest.TestCase`，文件命名为 `test_<模块>_<功能>.py`，测试方法以 `test_` 开头。修复缺陷时增加可复现问题的回归测试，数值测试使用确定性数据。先运行受影响模块，再运行全量测试；目前没有强制覆盖率门槛。JYDB 测试需要 PostgreSQL 配置，部分集成测试依赖网络或本地数据，提交时说明未运行的检查及原因。

## 文档与 Skill 维护
`docs/` 中的 Notebook 是文档和 `quantstudio` Skill 参考资料的唯一事实来源；先阅读 `docs/通则和约定.ipynb`。不要直接修改生成的 `docs_md/`、`site/` 或 `skills/quantstudio/references/`。

修改 Notebook 时，为真实 API 示例的代码单元维护 `skill` 标签；运行 `python scripts/build_skill.py --output-dir .claude/skills` 重新生成参考资料，并检查链接。Skill 仅保留框架知识，环境相关的表名、字段和路径通过 MCP 查询或由用户指定。

## 提交与拉取请求
历史提交混用简短描述及 `fix(scripts):`、`feat(scripts):` 等前缀；建议使用明确的作用域和变更说明，每次提交聚焦一个目的。拉取请求说明问题、修改后的行为、验证结果及配置前提；关联已有 issue，涉及可视化变更时附截图。

## 配置与安全
参考 `examples/QuantStudioConfig/`，本地配置通常存放于 `~/QuantStudioConfig/`。不要硬编码凭据或将密钥、机器路径、`.env`、本地 `.mcp.json` 和测试数据提交到仓库。参数优先采用显式传入值，再读取配置文件，最后使用模型默认值。
