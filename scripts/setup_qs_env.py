#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""QuantStudio 环境配置脚本。

把 QuantStudio 源码仓库中的模板与 Skill 配置到一个目标目录下，产出一份可直接
使用的开发环境。所有内容均取自脚本所在的 QuantStudio 仓库（源码模式），
不需要预先安装 QuantStudio 包。

写入目标目录（`--target-dir`，默认为本仓库根目录）的内容：

    .mcp.json                     MCP 服务配置
    CLAUDE.md                     QS 框架概述（各项目共用）
    CLAUDE.local.md               本地约定 + 可用因子库/风险库说明（个人文件，通常不入库）
    .claude/skills/*              各 SKILL 的完整拷贝（非链接，可脱离本仓库使用）
    tests/ scripts/               目录约定（已在 CLAUDE.local.md 中声明）
    requirements.txt              依赖清单（拷自本仓库）
    examples/*.py                 可运行示例（拷自本仓库）
    docs/                         Notebook 文档（拷贝时排除 data/ 等运行产物）

配置完成后默认执行自检：QS 可导入、各因子库与风险库可连接、MCP 服务可启动。失败只告警
不中断——因子库不可达是常见状态，不应阻断环境配置（--no-check 可跳过）。

脚本还会检查 ~/QuantStudioConfig/ 下各因子库、风险库配置是否齐备，缺失时列出所需字段。
该目录中的连接信息含凭据，脚本只提示不代写。

使用方法:
    # 配置本仓库自身（默认：--target-dir 为脚本所在仓库根）
    python scripts/setup_qs_env.py

    # 把 QS 环境配置到指定目录
    python scripts/setup_qs_env.py --target-dir D:/Project/MyProject

    # 跳过依赖安装（依赖已就绪时）
    python scripts/setup_qs_env.py --skip-pip

    # 指定 MCP 文档缓存目录（默认 D:/Data/JYDBDoc）
    python scripts/setup_qs_env.py --cache-dir E:/Cache/JYDBDoc

    # 覆盖已存在的 .mcp.json / CLAUDE.md / CLAUDE.local.md / skill 链接与生成产物
    python scripts/setup_qs_env.py --force

    # 仅预览将要执行的操作，不做任何改动
    python scripts/setup_qs_env.py --dry-run

    # 跳过配置完成后的自检
    python scripts/setup_qs_env.py --no-check

    # 不生成 Python 工程骨架（只配 Claude Code 相关文件）
    python scripts/setup_qs_env.py --skip-skeleton

    # 指定 pip 使用的镜像源
    python scripts/setup_qs_env.py --pip-index-url https://pypi.tuna.tsinghua.edu.cn/simple

    # 指定因子库配置（TYPE 取 JYDB/HDF5DB/SQLDB/BaoStockDB，可重复；同类型可传多份配置）
    python scripts/setup_qs_env.py --factor-db HDF5DB=D:/MyData/HDF5DBConfig.json
    python scripts/setup_qs_env.py \
        --factor-db JYDB=D:/Conf/JYDBConfig.json \
        --factor-db HDF5DB=D:/MyData/HDF5DBConfig.json

    # 指定风险库配置（TYPE 取 HDF5FRDB/HDF5RDB，规则同上）
    python scripts/setup_qs_env.py --risk-db HDF5FRDB=D:/MyRisk/HDF5FRDBConfig.json

因子库配置会写入 CLAUDE.local.md 的「可用因子库」章节，风险库配置写入「可用风险库」
章节。未通过 --factor-db / --risk-db 指定的类型，脚本会在 ~/QuantStudioConfig/ 下
查找其默认配置（如 JYDBConfig.json），存在则一并写入；显式传入的配置与默认配置会
各占一节。

前置要求:
    - 建议使用 QS 环境的解释器执行本脚本（依赖安装的目标即该解释器）；
      若解释器路径不是虚拟环境（无 pyvenv.cfg），脚本会提示。
    - 目标目录不是 git 仓库时，脚本会对其执行 git init；需要 PATH 中有 git。
    - 生成 skill 链接时，Windows 下符号链接需开发者模式或管理员权限；
      权限不足时会自动回退为目录联接。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_SRC = REPO_ROOT / "skills"
TEMPLATE_MCP = REPO_ROOT / ".mcp.example.json"
TEMPLATE_CLAUDE = REPO_ROOT / "CLAUDE.local.example.md"
BUILD_SKILL = REPO_ROOT / "scripts" / "build_skill.py"
DEFAULT_CACHE_DIR = "D:/Data/JYDBDoc"

# 已知因子库类型 -> (默认配置文件名, 章节标题, 内容描述)
FACTOR_DB_TYPES = {
    "JYDB": ("JYDBConfig.json", "JYDB（聚源数据库）", "股票、基金等证券的基本信息、行情、财务等金融数据"),
    "HDF5DB": ("HDF5DBConfig.json", "HDF5DB（本地 HDF5 因子库）", "自算因子与衍生数据的本地存储，读写均可"),
    "SQLDB": ("SQLDBConfig.json", "SQLDB（通用 SQL 因子库）", "用户自建的 SQL 数据库，读写均可"),
    "BaoStockDB": ("BaoStockDBConfig.json", "BaoStockDB（BaoStock 数据源）", "股票与指数的行情、财务数据"),
}
# 需要额外附加说明的类型 -> 提示文字
FACTOR_DB_NOTES = {
    "BaoStockDB": "**注意**：本库仅用于测试，不要用于生产。",
}
# QS 不可导入时无法通过字段声明识别敏感字段，退化为按名称排除
FALLBACK_SECRET_FIELDS = {"Pwd"}

# 各因子库的默认配置文件名与需要的字段，用于在 ~/QuantStudioConfig/ 缺失时给出提示
CONFIG_TEMPLATE_HINTS = {
    "JYDB": ("JYDBConfig.json", "DBType / DBName / IPAddr / Port / User / Pwd"),
    "HDF5DB": ("HDF5DBConfig.json", "MainDir（存放因子数据的本地目录）"),
    "SQLDB": ("SQLDBConfig.json", "DBType / DBName / IPAddr / Port / User / Pwd"),
    "BaoStockDB": ("BaoStockDBConfig.json", "通常只需 {\"Name\": \"BaoStockDB\"}，登录走匿名 API"),
}

# 已知风险库类型 -> (默认配置文件名, 章节标题, 内容描述)
RISK_DB_TYPES = {
    "HDF5FRDB": ("HDF5FRDBConfig.json", "HDF5FRDB（多因子风险库）",
                 "结构化多因子风险数据：因子暴露、因子协方差、特异性风险与收益率"),
    "HDF5RDB": ("HDF5RDBConfig.json", "HDF5RDB（本地风险库）",
                "直接的协方差矩阵，不分解为因子模型"),
}

# 各风险库的默认配置文件名与需要的字段，用于在 ~/QuantStudioConfig/ 缺失时给出提示
RISK_CONFIG_TEMPLATE_HINTS = {
    "HDF5FRDB": ("HDF5FRDBConfig.json", "MainDir（存放风险数据的本地目录）"),
    "HDF5RDB": ("HDF5RDBConfig.json", "MainDir（存放风险数据的本地目录）"),
}

# 目标目录 CLAUDE.md 的正文。项目约定、运行环境与数据库由 CLAUDE.local.md 承担，
# 此处只保留 QS 框架概述，避免两处重复。
CLAUDE_MD_BODY = """\
# 量化研究框架

本项目使用 QuantStudio v2.0.0（Python 3.12+）。因子计算、回测、风险建模、组合优化
均构建在自研计算图引擎之上，每个量化操作被建模为由 Node 组成的有向无环图（DAG）。

完整的 API 用法与示例见项目 Skill：`.claude/skills/quantstudio/`（SKILL.md 为索引，
references/ 下为各模块文档），也可通过 `docs/` 下的 Jupyter Notebook 查阅。
"""

# 文档派生型 Skill：其 references/ 由 docs/ 生成（见 CLAUDE.md「Skill 维护」），
# 因此不能链接——链接只会得到缺少 references/ 的空壳，必须用 build_skill.py 生成。
GENERATED_SKILLS = {"quantstudio"}


def _log(msg: str, *, level: str = "INFO") -> None:
    prefix = {"INFO": "[INFO]", "WARN": "[WARN]", "SKIP": "[SKIP]", "DRY": "[DRY-RUN]"}[level]
    print(f"{prefix} {msg}")


def _resolve_dir(raw: str) -> Path:
    return Path(raw).expanduser().resolve()


def install_requirements(python: str, dry_run: bool, index_url: str | None) -> None:
    """在当前解释器中安装本仓库 requirements.txt 中的依赖。"""
    req = REPO_ROOT / "requirements.txt"
    if not req.exists():
        raise FileNotFoundError(f"未找到依赖清单: {req}")

    cmd = [python, "-m", "pip", "install", "-r", str(req)]
    if index_url:
        cmd += ["-i", index_url]

    if dry_run:
        _log(f"将执行: {' '.join(cmd)}", level="DRY")
        return

    _log(f"安装依赖: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def write_mcp_config(target_dir: Path, python: str, cache_dir: str, force: bool, dry_run: bool) -> None:
    """渲染 .mcp.example.json 生成目标目录下的 .mcp.json。"""
    target = target_dir / ".mcp.json"
    if target.exists() and not force:
        _log(f"{target} 已存在，跳过（--force 可覆盖）", level="SKIP")
        return

    text = TEMPLATE_MCP.read_text(encoding="utf-8")
    replacements = {
        "{{PYTHON}}": python,
        "{{SERVER_PY}}": (REPO_ROOT / "mcp" / "jy_doc" / "server.py").as_posix(),
        "{{PROJECT}}": target_dir.as_posix(),
        "{{CACHE_DIR}}": _resolve_dir(cache_dir).as_posix(),
        "{{PYTHONPATH}}": REPO_ROOT.as_posix(),
    }
    for key, value in replacements.items():
        text = text.replace(key, value.replace("\\", "\\\\"))

    # 校验替换后仍是合法 JSON；残留占位符会让解析失败或漏改字段
    if "{{" in text:
        leftover = text[text.index("{{") : text.index("{{") + 40]
        raise ValueError(f"模板中存在未替换的占位符: {leftover}")
    json.loads(text)

    if dry_run:
        _log(f"将写入 {target}", level="DRY")
        print(text)
        return

    target.write_text(text, encoding="utf-8")
    _log(f"已生成 {target}")


def copy_skills(target_dir: Path, force: bool, dry_run: bool) -> None:
    """把本仓库 skills/ 下的每个 SKILL 拷到目标目录的 .claude/skills/。

    文档派生型 Skill（见 GENERATED_SKILLS）不在此处理，由 build_generated_skills 生成。
    """
    if not SKILLS_SRC.is_dir():
        raise FileNotFoundError(f"未找到 skill 源目录: {SKILLS_SRC}")

    sources = sorted(
        p for p in SKILLS_SRC.iterdir()
        if (p / "SKILL.md").is_file() and p.name not in GENERATED_SKILLS
    )
    if not sources:
        _log(f"{SKILLS_SRC} 下没有找到含 SKILL.md 的目录", level="WARN")
        return

    dst_root = target_dir / ".claude" / "skills"
    if not dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)

    for src in sources:
        dst = dst_root / src.name

        if dst.exists():
            if not force:
                _log(f"{dst} 已存在，跳过（--force 可覆盖）", level="SKIP")
                continue
            if dry_run:
                _log(f"将删除后重新拷贝: {dst}", level="DRY")
            else:
                shutil.rmtree(dst)

        if dry_run:
            _log(f"将拷贝 Skill: {src.name} -> {dst}", level="DRY")
            continue

        shutil.copytree(src, dst)
        _log(f"已拷贝 Skill: {src.name} -> {dst}")


def build_generated_skills(target_dir: Path, force: bool, dry_run: bool) -> None:
    """用 build_skill.py 生成文档派生型 Skill 到目标目录的 .claude/skills/。

    内容取自本仓库的 docs/ 与 mkdocs.yml（源码模式），与 --target-dir 指向何处无关。
    """
    if not GENERATED_SKILLS:
        return
    if not BUILD_SKILL.is_file():
        raise FileNotFoundError(f"未找到 Skill 生成脚本: {BUILD_SKILL}")

    dst_root = target_dir / ".claude" / "skills"

    for name in sorted(GENERATED_SKILLS):
        dst = dst_root / name
        cmd = [sys.executable, str(BUILD_SKILL), "--output-dir", str(dst_root)]

        if dry_run:
            if dst.exists() and force:
                _log(f"将删除后重新生成 Skill: {dst}", level="DRY")
            _log(f"将生成 Skill: {' '.join(cmd)}", level="DRY")
            continue

        if dst.exists():
            if not force:
                _log(f"{dst} 已存在，跳过（--force 可删除重建）", level="SKIP")
                continue
            # build_skill.py 的 --clean 是终止性的（清理后即退出），故在此先行删除
            shutil.rmtree(dst)
            _log(f"已删除旧产物: {dst}")

        _log(f"生成 Skill {name}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def write_claude_md(target_dir: Path, force: bool, dry_run: bool) -> None:
    """生成目标目录下的 CLAUDE.md（QS 框架概述）。"""
    target = target_dir / "CLAUDE.md"
    if target.exists() and not force:
        _log(f"{target} 已存在，跳过（--force 可覆盖）", level="SKIP")
        return

    if dry_run:
        _log(f"将写入 {target}", level="DRY")
        print(CLAUDE_MD_BODY)
        return

    target.write_text(CLAUDE_MD_BODY, encoding="utf-8")
    _log(f"已生成 {target}")


def _secret_fields(db_type: str) -> set:
    """返回 db_type 对应的敏感字段名集合。

    优先读取 QS 字段声明中 json_schema_extra={"secret": True} 的标记，这样新增
    因子库类型或敏感字段时无需改本脚本；QS 不可导入时退化为按名称排除。
    """
    try:
        from QuantStudio.Factor import api as _factor_api
    except Exception:
        _log("无法导入 QuantStudio，敏感字段退化为按名称排除", level="WARN")
        return set(FALLBACK_SECRET_FIELDS)

    cls = getattr(_factor_api, db_type, None)
    arg_cls = getattr(cls, "__QS_ArgClass__", None)
    fields = getattr(arg_cls, "model_fields", None)
    if not fields:
        return set(FALLBACK_SECRET_FIELDS)
    return {
        name for name, field in fields.items()
        if (getattr(field, "json_schema_extra", None) or {}).get("secret")
    }


def _default_config_path(db_type: str) -> Path:
    """该类型在 ~/QuantStudioConfig/ 下的默认配置文件路径。"""
    table = FACTOR_DB_TYPES if db_type in FACTOR_DB_TYPES else RISK_DB_TYPES
    return Path(os.path.expanduser("~")) / "QuantStudioConfig" / table[db_type][0]


def _ctor(db_type: str, config_path: Path, cls: str) -> str:
    """构造该库的调用表达式。

    配置文件不是默认路径时必须显式传入，否则对象会按默认配置创建，连到别的库。
    """
    if config_path.resolve() == _default_config_path(db_type).resolve():
        return f"{cls}()"
    return f"{cls}(config_file=r'{config_path}')"


def _db_example(db_type: str, config_path: Path) -> str:
    """返回该类型在指定配置文件下的读取示例代码块。"""
    if db_type == "JYDB":
        ctor = _ctor(db_type, config_path, "JYDB")
        return f'''\
```python
import datetime as dt
from QuantStudio.Factor.api import JYDB

db = {ctor}
db.connect()
db.TableNames                          # 可用表名列表
db.getStockID(exchange=("SSE", "SZSE"), date=dt.datetime(2024, 6, 28))   # 证券 ID
db.getTradeDay(start_date=dt.datetime(2024, 6, 24), end_date=dt.datetime(2024, 6, 28))

tbl = db.getTable("A股证券主表")
tbl.FactorNames                        # 该表的因子名
tbl.getFactor("证券代码").readData(ids, dts)      # -> DataFrame（时序 x 证券）
tbl.readData(["证券代码", "中文名称"], ids, dts)   # -> Panel（因子 x 时序 x 证券）
```
'''
    if db_type == "HDF5DB":
        ctor = _ctor(db_type, config_path, "HDF5DB")
        return f'''\
```python
from QuantStudio.Factor.api import HDF5DB

db = {ctor}
db.connect()
db.TableNames                          # 可用表名列表

tbl = db.getTable("stock_cn_info")
tbl.FactorNames                        # 该表的因子名
tbl.getID()                            # 证券 ID 列表
tbl.getDateTime()                      # 日期列表
tbl.readData(["name", "abbr"], ids, dts)   # -> Panel（因子 x 时序 x 证券）
```
'''
    if db_type == "BaoStockDB":
        ctor = _ctor(db_type, config_path, "BaoStockDB")
        return f'''\
```python
import datetime as dt
from QuantStudio.Factor.api import BaoStockDB

db = {ctor}
db.connect()
db.TableNames                          # 可用表名列表
db.getTradeDay(start_date=dt.datetime(2024, 6, 24), end_date=dt.datetime(2024, 6, 28))
db.getStockID(date=dt.datetime(2024, 6, 28))    # 证券 ID

tbl = db.getTable("每日A股K线")
tbl.FactorNames                        # 该表的因子名
tbl.readData(["close", "open"], ids, dts)   # -> Panel（因子 x 时序 x 证券）
```
'''
    ctor = _ctor(db_type, config_path, "SQLDB")
    return f'''\
```python
from QuantStudio.Factor.SQLDB import SQLDB

db = {ctor}
db.connect()
db.TableNames                          # 可用表名列表

tbl = db.getTable("<表名>")
tbl.FactorNames                        # 该表的因子名
tbl.readData(factor_names, ids, dts)   # -> Panel（因子 x 时序 x 证券）
```
'''


def render_factor_db_sections(specs: list[tuple[str, Path]]) -> str:
    """渲染「可用因子库」章节。specs 为 (类型, 配置文件路径) 列表，按传入顺序渲染。"""
    if not specs:
        return ""

    blocks = ["# 可用因子库"]
    for db_type, config_path in specs:
        _, title, description = FACTOR_DB_TYPES[db_type]
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"解析因子库配置失败 {config_path}: {e}")

        secret = _secret_fields(db_type)
        info = {k: v for k, v in config.items() if k not in secret}

        blocks.append(f"\n## {title}\n")
        blocks.append(f"- 配置文件：{config_path}")
        if db_type in ("JYDB", "SQLDB"):
            blocks.append(f"- 类型：{info.get('DBType', '')} / 库名 {info.get('DBName', '')}")
            blocks.append(f"- 服务器：{info.get('IPAddr', '')}:{info.get('Port', '')} / 用户 {info.get('User', '')}")
        elif db_type == "HDF5DB":
            blocks.append(f"- 主目录：{info.get('MainDir', '')}")
        blocks.append(f"- 内容：{description}")
        if db_type == "JYDB":
            blocks.append("- 表结构说明使用 `jy_doc` MCP 工具检索")
        if db_type in FACTOR_DB_NOTES:
            blocks.append(f"- {FACTOR_DB_NOTES[db_type]}")
        blocks.append("\n" + _db_example(db_type, config_path))

    return "\n".join(blocks) + "\n"


def resolve_factor_db_specs(raw_specs: list[str] | None) -> list[tuple[str, Path]]:
    """解析 --factor-db 参数，并在末尾追加各类型的默认配置（文件存在才加）。

    每个类型都尝试默认路径 ~/QuantStudioConfig/<TYPE>Config.json；显式传入的排在
    兜底之前，因此同类型可以出现多节（如多个 HDF5 目录各有一份配置）。
    """
    specs: list[tuple[str, Path]] = []
    for raw in raw_specs or []:
        if "=" not in raw:
            raise ValueError(f"--factor-db 需要 TYPE=PATH 形式，收到: {raw}")
        db_type, _, path = raw.partition("=")
        db_type = db_type.strip()
        if db_type not in FACTOR_DB_TYPES:
            raise ValueError(f"未知因子库类型 '{db_type}'，已知类型: {', '.join(sorted(FACTOR_DB_TYPES))}")
        config_path = _resolve_dir(path.strip())
        if not config_path.is_file():
            _log(f"未找到 {db_type} 的配置 {config_path}，跳过", level="SKIP")
            continue
        specs.append((db_type, config_path))

    config_dir = Path(os.path.expanduser("~")) / "QuantStudioConfig"
    for db_type, (filename, _, _) in FACTOR_DB_TYPES.items():
        default = config_dir / filename
        if default.is_file():
            specs.append((db_type, default))
        else:
            _log(f"未找到 {db_type} 的默认配置 {default}，跳过", level="SKIP")
    return specs


def _risk_db_example(db_type: str, config_path: Path) -> str:
    """返回该风险库在指定配置文件下的读取示例代码块。"""
    if db_type == "HDF5FRDB":
        ctor = _ctor(db_type, config_path, "HDF5FRDB")
        return f'''\
```python
import datetime as dt
from QuantStudio.Risk.HDF5RDB import HDF5FRDB

RDB = {ctor}
RDB.connect()
RDB.TableNames                          # 风险表名列表

RT = RDB.getTable("<表名>")
RT.FactorNames                          # 因子名
DTs = RT.getDateTime()                  # 时点序列
IDs = RT.getID(idt=DTs[-1])             # 某时点的证券 ID

RT.readFactorData(dts=DTs, ids=IDs)     # 因子暴露 -> Panel（因子 x 时点 x 证券）
RT.readFactorCov(dts=DTs)               # 因子协方差 -> Panel（因子 x 因子 x 时点）
RT.readSpecificRisk(dts=DTs, ids=IDs)   # 特异性风险 -> DataFrame（时点 x 证券）
RT.readFactorReturn(dts=DTs)            # 因子收益率 -> DataFrame（时点 x 因子）
```
'''
    ctor = _ctor(db_type, config_path, "HDF5RDB")
    return f'''\
```python
import datetime as dt
from QuantStudio.Risk.HDF5RDB import HDF5RDB

RDB = {ctor}
RDB.connect()
RDB.TableNames                          # 风险表名列表

RT = RDB.getTable("<表名>")
DTs = RT.getDateTime()                  # 时点序列
IDs = RT.getID(idt=DTs[-1])             # 某时点的证券 ID
RT.readCov(dts=DTs, ids=IDs)            # 协方差矩阵 -> Panel（证券 x 证券 x 时点）
```
'''


def render_risk_db_sections(specs: list[tuple[str, Path]]) -> str:
    """渲染「可用风险库」章节。specs 为 (类型, 配置文件路径) 列表，按传入顺序渲染。"""
    if not specs:
        return ""

    blocks = ["# 可用风险库"]
    for db_type, config_path in specs:
        _, title, description = RISK_DB_TYPES[db_type]
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"解析风险库配置失败 {config_path}: {e}")

        secret = _secret_fields(db_type)
        info = {k: v for k, v in config.items() if k not in secret}

        blocks.append(f"\n## {title}\n")
        blocks.append(f"- 配置文件：{config_path}")
        blocks.append(f"- 主目录：{info.get('MainDir', '')}")
        blocks.append(f"- 内容：{description}")
        blocks.append("\n" + _risk_db_example(db_type, config_path))

    return "\n".join(blocks) + "\n"


def resolve_risk_db_specs(raw_specs: list[str] | None) -> list[tuple[str, Path]]:
    """解析 --risk-db 参数，并在末尾追加各类型的默认配置（文件存在才加）。"""
    specs: list[tuple[str, Path]] = []
    for raw in raw_specs or []:
        if "=" not in raw:
            raise ValueError(f"--risk-db 需要 TYPE=PATH 形式，收到: {raw}")
        db_type, _, path = raw.partition("=")
        db_type = db_type.strip()
        if db_type not in RISK_DB_TYPES:
            raise ValueError(f"未知风险库类型 '{db_type}'，已知类型: {', '.join(sorted(RISK_DB_TYPES))}")
        config_path = _resolve_dir(path.strip())
        if not config_path.is_file():
            _log(f"未找到 {db_type} 的配置 {config_path}，跳过", level="SKIP")
            continue
        specs.append((db_type, config_path))

    config_dir = Path(os.path.expanduser("~")) / "QuantStudioConfig"
    for db_type, (filename, _, _) in RISK_DB_TYPES.items():
        default = config_dir / filename
        if default.is_file():
            specs.append((db_type, default))
        else:
            _log(f"未找到 {db_type} 的默认配置 {default}，跳过", level="SKIP")
    return specs


def write_claude_local(
    target_dir: Path, python: str, db_specs: list[tuple[str, Path]],
    risk_specs: list[tuple[str, Path]], force: bool, dry_run: bool
) -> None:
    """渲染 CLAUDE.local.example.md 生成目标目录下的 CLAUDE.local.md。

    可用因子库、可用风险库章节分别由 db_specs、risk_specs 渲染后追加在模板内容之后。
    """
    target = target_dir / "CLAUDE.local.md"
    if target.exists() and not force:
        _log(f"{target} 已存在，跳过（--force 可覆盖）", level="SKIP")
        return

    text = TEMPLATE_CLAUDE.read_text(encoding="utf-8")
    lines = []
    replaced = False
    for line in text.splitlines(keepends=True):
        if not replaced and line.lstrip().startswith("Python："):
            lines.append(f"Python：使用 QS 环境，解释器位置是：{python}\n")
            replaced = True
        else:
            lines.append(line)

    if not replaced:
        _log("模板中未匹配到 Python 路径行，原样复制", level="WARN")
        lines = [text]

    rendered = "".join(lines)
    for sections in (render_factor_db_sections(db_specs), render_risk_db_sections(risk_specs)):
        if sections:
            rendered = rendered.rstrip("\n") + "\n\n" + sections

    if dry_run:
        _log(f"将写入 {target}", level="DRY")
        print(rendered)
        return

    target.write_text(rendered, encoding="utf-8")
    _log(f"已生成 {target}")


def ensure_git_repo(target_dir: Path, dry_run: bool) -> None:
    """目标目录未被任何 git 仓库覆盖时，对其执行 git init。

    目标目录已是仓库根，或位于某个父级仓库内时均跳过——git 不支持嵌套仓库，
    在已有仓库覆盖的目录里再 init 会造出父仓库无法跟踪、也看不到历史的嵌入式仓库。
    """
    if (target_dir / ".git").exists():
        _log(f"{target_dir} 已是 git 仓库，跳过", level="SKIP")
        return

    result = subprocess.run(
        ["git", "-C", str(target_dir), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        toplevel = result.stdout.strip()
        if toplevel and Path(toplevel).resolve() != target_dir.resolve():
            _log(f"{target_dir} 已属于仓库 {toplevel}，跳过", level="SKIP")
            return

    if dry_run:
        _log(f"将执行: git init {target_dir}", level="DRY")
        return

    _log(f"初始化 git 仓库: {target_dir}")
    subprocess.run(["git", "init", str(target_dir)], check=True)


def _write_file(target: Path, content: str, force: bool, dry_run: bool) -> None:
    """写入单个文件，已存在且未指定 --force 时跳过。"""
    if target.exists() and not force:
        _log(f"{target} 已存在，跳过（--force 可覆盖）", level="SKIP")
        return
    if dry_run:
        _log(f"将写入 {target}", level="DRY")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    _log(f"已生成 {target}")


def make_project_skeleton(target_dir: Path, force: bool, dry_run: bool) -> None:
    """在目标目录建立目录约定（tests/ 与 scripts/）并拷入依赖清单。

    环境是作为工作目录使用，而非待发布的 Python 包，因此不生成包目录与
    pyproject.toml——需要打包的用户可自行创建。
    """
    for sub in ("tests", "scripts"):
        path = target_dir / sub
        if dry_run:
            _log(f"将创建目录 {path}", level="DRY")
        elif not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            _log(f"已创建目录 {path}")
        else:
            _log(f"{path} 已存在，跳过", level="SKIP")

    _write_file(target_dir / "requirements.txt", (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8"),
                force, dry_run)


def copy_examples(target_dir: Path, force: bool, dry_run: bool) -> None:
    """把本仓库 examples/ 下的可运行示例拷到目标目录，作为入门材料。"""
    src = REPO_ROOT / "examples"
    if not src.is_dir():
        _log(f"未找到示例目录 {src}，跳过", level="SKIP")
        return

    dst = target_dir / "examples"
    files = sorted(p for p in src.iterdir() if p.suffix == ".py")
    if not files:
        _log(f"{src} 下没有 .py 示例，跳过", level="SKIP")
        return

    if dst.exists() and not force:
        _log(f"{dst} 已存在，跳过（--force 可覆盖）", level="SKIP")
        return

    if dry_run:
        _log(f"将拷贝 {len(files)} 个示例到 {dst}", level="DRY")
        return

    dst.mkdir(parents=True, exist_ok=True)
    for path in files:
        shutil.copy2(path, dst / path.name)
    _log(f"已拷贝 {len(files)} 个示例到 {dst}")


# 拷贝 docs/ 时排除的目录：data/ 是 notebook 的运行产物（已 gitignore），
# 检查点与字节码缓存属于编辑/运行残留，均非文档内容。
DOCS_EXCLUDE_DIRS = {"data", ".ipynb_checkpoints", "__pycache__"}


def copy_docs(target_dir: Path, force: bool, dry_run: bool) -> None:
    """把本仓库 docs/ 下的 Notebook 文档拷到目标目录，供查阅完整 API 用法。"""
    src = REPO_ROOT / "docs"
    if not src.is_dir():
        _log(f"未找到文档目录 {src}，跳过", level="SKIP")
        return

    dst = target_dir / "docs"
    if dst.exists() and not force:
        _log(f"{dst} 已存在，跳过（--force 可覆盖）", level="SKIP")
        return

    def ignore(directory, names):
        return [n for n in names if n in DOCS_EXCLUDE_DIRS]

    count = sum(1 for p in src.rglob("*") if p.is_file() and not (DOCS_EXCLUDE_DIRS & set(p.parts)))
    if dry_run:
        _log(f"将拷贝 {count} 个文档文件到 {dst}", level="DRY")
        return

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)
    _log(f"已拷贝 {count} 个文档文件到 {dst}")


def check_config_dir(dry_run: bool) -> None:
    """检查 ~/QuantStudioConfig/ 下各因子库、风险库配置文件，缺失时给出建立提示。"""
    config_dir = Path(os.path.expanduser("~")) / "QuantStudioConfig"
    hints = {**CONFIG_TEMPLATE_HINTS, **RISK_CONFIG_TEMPLATE_HINTS}
    missing = [
        (db_type, filename, hint)
        for db_type, (filename, hint) in hints.items()
        if not (config_dir / filename).is_file()
    ]
    if not missing:
        _log(f"{config_dir} 下的因子库、风险库配置齐备", level="SKIP")
        return

    if dry_run:
        _log(f"将检查配置目录 {config_dir}（当前缺失 {len(missing)} 个文件）", level="DRY")
        return

    _log(f"{config_dir} 下缺失 {len(missing)} 个配置文件：", level="WARN")
    for db_type, filename, hint in missing:
        print(f"        {config_dir / filename}")
        print(f"            ({db_type} 需要字段：{hint})")
    print("        连接信息请向数据提供方索取后手动创建，脚本不会代写。")


# 各库的导入路径。未被 Factor.api / Risk.api 导出的类需从其自身模块取
_DB_IMPORT_OVERRIDES = {
    "SQLDB": "from QuantStudio.Factor.SQLDB import SQLDB as _Cls",
    "HDF5RDB": "from QuantStudio.Risk.HDF5RDB import HDF5RDB as _Cls",
    "HDF5FRDB": "from QuantStudio.Risk.HDF5RDB import HDF5FRDB as _Cls",
}
_DB_MODULE_ROOT = {"HDF5RDB": "Risk", "HDF5FRDB": "Risk"}


def _db_probe(db_type: str, config_path: Path) -> tuple[str, Path, str, str]:
    """返回 (类型, 配置, import 语句, 类表达式)，供自检构造子进程探针。"""
    if db_type in _DB_IMPORT_OVERRIDES:
        return db_type, config_path, _DB_IMPORT_OVERRIDES[db_type], "_Cls"
    root = _DB_MODULE_ROOT.get(db_type, "Factor")
    return db_type, config_path, f"from QuantStudio.{root} import api as _api", f"_api.{db_type}"


def run_self_check(
    target_dir: Path, python: str,
    db_specs: list[tuple[str, Path]], risk_specs: list[tuple[str, Path]]
) -> None:
    """配置完成后自检：QS 可导入、各因子库/风险库可连接、MCP 可启动。

    任一检查失败只告警不中断——数据源不可达是常见状态，不应阻断环境配置。
    """
    print("-" * 60)
    _log("开始自检")

    # 1) QS 可导入
    probe = (
        "import sys; sys.path.insert(0, %r); import QuantStudio; "
        "print('QS', QuantStudio.__file__)" % REPO_ROOT.as_posix()
    )
    try:
        result = subprocess.run([python, "-c", probe], capture_output=True, text=True,
                                cwd=str(target_dir), timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        _log(f"QuantStudio 导入检查失败: {e}", level="WARN")
    else:
        if result.returncode == 0:
            _log(f"QuantStudio 可导入: {result.stdout.strip()}")
        else:
            _log(f"QuantStudio 导入失败: {result.stderr.strip().splitlines()[-1] if result.stderr else '未知错误'}", level="WARN")

    # 2) 各库可连接。Factor.api / Risk.api 未导出全部类，需按类型指定导入路径
    probes = [(_db_probe(t, p)) for t, p in db_specs] + [_db_probe(t, p) for t, p in risk_specs]
    for db_type, config_path, import_stmt, cls_expr in probes:
        probe = (
            "import sys; sys.path.insert(0, %r); %s; "
            "db = %s(config_file=%r); db.connect(); "
            "print(len(db.TableNames))" % (REPO_ROOT.as_posix(), import_stmt, cls_expr, config_path.as_posix())
        )
        # 超时或进程起不来按连接失败处理：数据源不可达是常见状态，
        # 不能让它中断自检（否则整个环境配置都会失败）
        try:
            result = subprocess.run([python, "-c", probe], capture_output=True, text=True,
                                    cwd=str(target_dir), timeout=120)
        except (OSError, subprocess.TimeoutExpired) as e:
            _log(f"{db_type} 连接失败（{config_path}）: {e}", level="WARN")
            continue
        if result.returncode == 0:
            _log(f"{db_type} 连接成功，{result.stdout.strip().splitlines()[-1]} 张表")
        else:
            last = result.stderr.strip().splitlines()[-1] if result.stderr else "未知错误"
            _log(f"{db_type} 连接失败（{config_path}）: {last}", level="WARN")

    # 3) MCP 可启动
    mcp_config = target_dir / ".mcp.json"
    if not mcp_config.is_file():
        _log("未找到 .mcp.json，跳过 MCP 检查", level="SKIP")
        return
    try:
        server = json.loads(mcp_config.read_text(encoding="utf-8"))["mcpServers"]["jy_doc"]
    except (KeyError, json.JSONDecodeError) as e:
        _log(f".mcp.json 解析失败: {e}", level="WARN")
        return

    request = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                   "clientInfo": {"name": "setup-check", "version": "1"}},
    })
    env = dict(os.environ, **server.get("env", {}))
    try:
        proc = subprocess.run(
            [server["command"], *server["args"]], input=request + "\n", capture_output=True,
            text=True, cwd=server.get("cwd", str(target_dir)), env=env, timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        _log(f"MCP 服务启动失败: {e}", level="WARN")
        return
    if '"result"' in proc.stdout:
        _log("MCP 服务 (jy_doc) 启动正常")
    else:
        _log(f"MCP 服务无有效响应: {proc.stderr.strip()[-200:] if proc.stderr else proc.stdout[:200]}", level="WARN")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把 QuantStudio 的模板与 Skill 配置到目标目录（安装依赖、生成 MCP 配置、链接 Skill、生成本地约定文件）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target-dir",
        default=None,
        help="配置写入的目标目录；未指定时使用本仓库根目录",
    )
    parser.add_argument("--skip-pip", action="store_true", help="跳过依赖安装")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help=f"聚源文档缓存目录 (默认: {DEFAULT_CACHE_DIR})")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的 .mcp.json / CLAUDE.local.md / skill 链接与生成产物")
    parser.add_argument("--dry-run", action="store_true", help="仅预览操作，不做任何改动")
    parser.add_argument("--pip-index-url", default=None, help="pip 镜像源地址")
    parser.add_argument("--no-check", action="store_true", help="配置完成后跳过自检")
    parser.add_argument("--skip-skeleton", action="store_true", help="不生成 Python 工程骨架")
    parser.add_argument(
        "--factor-db",
        action="append",
        default=None,
        metavar="TYPE=PATH",
        help=f"因子库配置，可重复。TYPE 取 {'/'.join(sorted(FACTOR_DB_TYPES))}；"
             "未指定的类型会在 ~/QuantStudioConfig/ 下查找默认配置",
    )
    parser.add_argument(
        "--risk-db",
        action="append",
        default=None,
        metavar="TYPE=PATH",
        help=f"风险库配置，可重复。TYPE 取 {'/'.join(sorted(RISK_DB_TYPES))}；"
             "未指定的类型会在 ~/QuantStudioConfig/ 下查找默认配置",
    )
    args = parser.parse_args()

    python = sys.executable
    if not python:
        _log("无法获取当前解释器路径 (sys.executable 为空)", level="WARN")
        return 1
    # 不能对解释器路径做 resolve()：venv 的 bin/python 是指向基础解释器的符号链接，
    # 解析后会丢失虚拟环境身份，导致依赖装到基础解释器里
    python = str(Path(python).absolute())

    if args.target_dir:
        target_dir = _resolve_dir(args.target_dir)
        if not target_dir.is_dir():
            raise NotADirectoryError(f"--target-dir 指定的目录不存在: {target_dir}")
    else:
        target_dir = REPO_ROOT
        _log(f"未指定 --target-dir，使用本仓库根目录: {target_dir}")

    print(f"仓库根目录: {REPO_ROOT}")
    print(f"配置目标目录: {target_dir}")
    print(f"Python 解释器: {python}")
    print(f"MCP 缓存目录: {_resolve_dir(args.cache_dir)}")
    print("-" * 60)

    if os.name == "nt" and not (Path(python).parent.parent / "pyvenv.cfg").exists():
        _log(f"{python} 看起来不是虚拟环境，依赖将装到该解释器的 site-packages", level="WARN")

    ensure_git_repo(target_dir, args.dry_run)

    if args.skip_pip:
        _log("按参数跳过依赖安装", level="SKIP")
    else:
        install_requirements(python, args.dry_run, args.pip_index_url)

    write_mcp_config(target_dir, python, args.cache_dir, args.force, args.dry_run)
    copy_skills(target_dir, args.force, args.dry_run)
    build_generated_skills(target_dir, args.force, args.dry_run)
    write_claude_md(target_dir, args.force, args.dry_run)
    db_specs = resolve_factor_db_specs(args.factor_db)
    risk_specs = resolve_risk_db_specs(args.risk_db)
    write_claude_local(target_dir, python, db_specs, risk_specs, args.force, args.dry_run)

    if args.skip_skeleton:
        _log("按参数跳过工程骨架生成", level="SKIP")
    else:
        make_project_skeleton(target_dir, args.force, args.dry_run)
    copy_examples(target_dir, args.force, args.dry_run)
    copy_docs(target_dir, args.force, args.dry_run)

    check_config_dir(args.dry_run)

    print("-" * 60)
    if args.dry_run:
        print("配置完成。（dry-run，未做任何改动）")
    elif args.no_check:
        print("配置完成。（已跳过自检）")
    else:
        print("配置完成。")
        run_self_check(target_dir, python, db_specs, risk_specs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
