# -*- coding: utf-8 -*-
"""聚源数据库(Gildata)文档抓取器。

负责从聚源数据字典平台(dd.gildata.com)获取表文档内容，
包括登录认证、目录树抓取、表详情获取等。支持本地 JSON 缓存。

使用方式:
    fetcher = JYDocFetcher(cache_dir="D:/Data/JYDBDoc", user="xxx", pwd="xxx")
    detail = fetcher.fetch_table_detail(258)
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_random

from .models import ColumnInfo, DatabaseInfo, TableDetail

logger = logging.getLogger(__name__)

REQUEST_DELAY = 1.0  # 请求间隔（秒），避免过快请求


class JYDocFetcher:
    """聚源数据库文档抓取器。

    Attributes:
        base_url: Gildata 数据字典平台基础 URL
        cache_dir: 文档内容缓存目录
        user: 登录用户名
        pwd: 登录密码
        session_id: 登录后的 SESSION cookie
        http_session: requests 会话对象
    """

    BASE_URL = "https://dd.gildata.com"

    def __init__(self, cache_dir: str = "", user: str = "", pwd: str = ""):
        """初始化抓取器。

        Args:
            cache_dir: 文档内容缓存目录路径，为空则不缓存。
            user: 登录用户名，默认从环境变量 JY_DOC_USER 读取
            pwd: 登录密码，默认从环境变量 JY_DOC_PWD 读取
        """
        self.cache_dir = cache_dir
        self.user = user or os.getenv("JY_DOC_USER", "")
        self.pwd = pwd or os.getenv("JY_DOC_PWD", "")
        self.session_id: Optional[str] = None
        self.http_session = requests.Session()
        # 登录与 SESSION 状态非线程安全：本对象在实际使用中为多线程共享的单例
        # （如 MCP 服务的 anyio 工作线程），并发登录会互相覆盖 SESSION cookie，
        # 故用锁串行化整个请求过程。
        self._lock = threading.Lock()
        self.http_session.headers.update({
            "Accept": "application/json, text/plain, */*",
            "Referer": f"{self.BASE_URL}/",
        })

        if self.cache_dir:
            os.makedirs(os.path.join(self.cache_dir, "tables"), exist_ok=True)
            os.makedirs(os.path.join(self.cache_dir, "trees"), exist_ok=True)

    # ── 认证管理 ──────────────────────────────────────────────────────

    def _ensure_session(self) -> str:
        """确保已登录，返回有效的 SESSION cookie。

        如果已有 session 则直接返回，否则执行登录流程。

        Returns:
            SESSION cookie 值

        Raises:
            RuntimeError: 登录失败时抛出
        """
        if self.session_id:
            return self.session_id

        if not self.user or not self.pwd:
            raise RuntimeError(
                "聚源文档平台登录凭据未配置。"
                "请设置环境变量 JY_DOC_USER 和 JY_DOC_PWD，"
                "或在初始化时传入 user 和 pwd 参数。"
            )

        self.session_id = self._login()
        logger.info("聚源文档平台登录成功")
        return self.session_id

    def _login(self) -> str:
        """执行登录流程，获取 SESSION cookie。

        Gildata 登录流程：
        1. GET /api/captcha 获取验证码图片和 SESSION cookie
        2. 使用 ddddocr 识别验证码
        3. POST /api/authentication 提交登录表单

        重试由调用方 _request 统一负责：此处若再加一层重试，会与 _request 的
        重试相乘（3×3 次），失败时耗时从约 10 秒放大到 30 秒以上。

        Returns:
            SESSION cookie 值
        """
        import datetime as dt

        # 清除残留的 SESSION：/api/captcha 仅在请求未携带 SESSION 时才下发新的
        # SESSION cookie，否则响应中不含 Set-Cookie 导致后续解析失败。
        self.http_session.cookies.clear()

        # 步骤 1：获取验证码
        now = dt.datetime.now().timestamp()
        rsp = self.http_session.get(
            url=f"{self.BASE_URL}/api/captcha?timestamp={int(now * 1000)}",
            headers={"Content-Type": "image/jpeg;charset=UTF-8"},
            cookies={"rememberMeFlag": "false"},
            timeout=30,
        )
        if rsp.status_code != 200:
            raise RuntimeError(f"验证码获取失败: status={rsp.status_code}")

        # 提取 SESSION
        session_id = None
        for cookie_val in rsp.headers.get("Set-Cookie", "").split(";"):
            parts = cookie_val.split("=")
            if parts[0].strip() == "SESSION":
                session_id = parts[-1].strip()
                break
        if not session_id:
            raise RuntimeError(f"SESSION 获取失败: {rsp.headers}")

        captcha_img = rsp.content

        # 步骤 2：识别验证码
        try:
            import ddddocr

            ocr = ddddocr.DdddOcr(det=False, ocr=True)
            captcha = ocr.classification(captcha_img)
        except ImportError:
            raise RuntimeError(
                "需要安装 ddddocr 库以识别验证码: pip install ddddocr"
            )

        # 步骤 3：登录
        login_rsp = self.http_session.post(
            url=(
                f"{self.BASE_URL}/api/authentication"
                f"?j_username={self.user}&j_password={self.pwd}"
                f"&j_captcha={captcha}&remember-me=false&submit=Login"
            ),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            cookies={"SESSION": session_id},
            timeout=30,
        )
        if login_rsp.status_code != 200:
            raise RuntimeError(
                f"登录失败: status={login_rsp.status_code}, body={login_rsp.text[:200]}"
            )

        # 更新 http_session 的 cookies
        self.http_session.cookies.set("SESSION", session_id)
        self.http_session.cookies.set("rememberMeFlag", "false")

        return session_id

    # ── HTTP 请求 ──────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_random(2, 4))
    def _request(self, url: str, check_empty: bool = True) -> Optional[dict | list]:
        """发送 GET 请求到 Gildata API。

        Args:
            url: 请求 URL
            check_empty: 是否检查返回值为空的情况

        Returns:
            解析后的 JSON 对象，失败时返回 None
        """
        with self._lock:
            self._ensure_session()

            time.sleep(random.uniform(0.5, REQUEST_DELAY))
            rsp = self.http_session.get(url=url, timeout=30)

            if rsp.status_code != 200:
                # session 可能过期，清除后重试
                self.session_id = None
                raise RuntimeError(
                    f"请求失败: url={url}, status={rsp.status_code}"
                )

        try:
            data = rsp.json()
        except Exception:
            if check_empty:
                raise RuntimeError(f"JSON 解析失败: url={url}")
            return None

        if check_empty and not data:
            raise RuntimeError(f"返回数据为空: url={url}")

        return data

    # ── 数据获取 ──────────────────────────────────────────────────────

    def get_database_list(self) -> list[DatabaseInfo]:
        """获取用户可访问的数据库列表。

        Returns:
            数据库信息列表
        """
        data = self._request(f"{self.BASE_URL}/api/DDUserPermission")
        if not data:
            return []

        databases = []
        for item in data:
            # API 返回格式: [id, ?, name, icon_url] 或 dict
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                databases.append(DatabaseInfo(id=int(item[0]), name=str(item[2])))
            elif isinstance(item, dict):
                databases.append(DatabaseInfo(
                    id=int(item.get("id", 0)),
                    name=str(item.get("name", item.get("groupName", ""))),
                ))

        logger.info("获取到 %d 个数据库", len(databases))
        return databases

    def fetch_tree(self, base_id: int) -> Optional[list[dict]]:
        """获取指定库的完整目录树。

        Args:
            base_id: 库 ID（product group ID）

        Returns:
            目录树的原始 JSON 结构（嵌套 dict 列表），失败时返回 None
        """
        # 检查缓存
        cache_path = self._tree_cache_path(base_id)
        if self.cache_dir and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        data = self._request(
            f"{self.BASE_URL}/api/productGroupTreeWithTables/{base_id}/-1/ALL_TREE",
            check_empty=False,
        )
        if not data:
            logger.warning("库 %d 目录树为空或无权限访问", base_id)
            return None

        # 写入缓存
        if self.cache_dir:
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except OSError as e:
                logger.warning("目录树缓存写入失败: %s", e)

        return data

    def fetch_table_detail(
        self, table_id: int, use_cache: bool = True, path: str = ""
    ) -> Optional[TableDetail]:
        """获取指定表的完整详情。

        调用多个 Gildata API 获取表信息、字段列表、从表字段、唯一索引等。

        Args:
            table_id: Gildata 平台中的表 ID
            use_cache: 是否使用本地缓存
            path: 文档路径（如果已知）

        Returns:
            表详情对象，失败时返回 None
        """
        # 检查缓存
        if use_cache:
            cached = self._load_table_cache(table_id)
            if cached is not None:
                logger.debug("使用缓存: table_id=%d", table_id)
                return cached

        # 获取表基本信息
        table_doc = self._request(f"{self.BASE_URL}/api/table/{table_id}")
        if not table_doc:
            return None

        # 检查权限
        table_data = table_doc.get("data", table_doc) if isinstance(table_doc, dict) else {}
        if table_data.get("tableAuth") == "NOT_HAS_AUTH":
            logger.warning("无权限访问表 %d", table_id)
            return None

        # 获取字段列表
        columns_raw = self._request(
            f"{self.BASE_URL}/api/column/{table_id}", check_empty=False
        )
        # 获取从表字段
        slave_columns = self._request(
            f"{self.BASE_URL}/api/slaveColumn/{table_id}", check_empty=False
        )
        # 获取唯一索引
        unique_index = self._request(
            f"{self.BASE_URL}/api/tableIndexByUnique/{table_id}", check_empty=False
        )

        # 解析字段列表
        columns = self._parse_columns(columns_raw if isinstance(columns_raw, list) else [])

        # 组装结果
        detail = TableDetail(
            table_id=table_id,
            table_name=table_data.get("tableChiName", ""),
            base_table_name=table_data.get("tableName", ""),
            path=path,
            description=table_data.get("description", ""),
            update_frequency=table_data.get("tableUpdateTime", ""),
            columns=columns,
            slave_columns=slave_columns if isinstance(slave_columns, list) else [],
            unique_index=unique_index if isinstance(unique_index, dict) else {},
            created_date=table_data.get("createdDate", ""),
            last_modified_date=table_data.get("lastModifiedDate", ""),
        )

        # 写入缓存
        if self.cache_dir:
            self._save_table_cache(table_id, detail)

        return detail

    def _parse_columns(self, raw_columns: list) -> list[ColumnInfo]:
        """解析原始字段数据为 ColumnInfo 列表。"""
        columns = []
        for col in raw_columns:
            if not isinstance(col, dict):
                continue
            # 跳过无效字段
            name = col.get("columnName", "")
            if not name:
                continue
            is_effective = col.get("isEffective", True)
            if not is_effective and is_effective is not None:
                continue

            columns.append(ColumnInfo(
                name=name,
                chinese_name=col.get("columnChiName", ""),
                data_type=col.get("columnType", ""),
                is_nullable=col.get("isNullable", "Y") == "Y",
                remark=col.get("remark", ""),
            ))

        return columns

    # ── 缓存管理 ──────────────────────────────────────────────────────

    def _tree_cache_path(self, base_id: int) -> str:
        """获取目录树缓存文件路径。"""
        return os.path.join(self.cache_dir, "trees", f"{base_id}.json")

    def _table_cache_path(self, table_id: int) -> str:
        """获取表详情缓存文件路径。"""
        return os.path.join(self.cache_dir, "tables", f"{table_id}.json")

    def _load_table_cache(self, table_id: int) -> Optional[TableDetail]:
        """从本地缓存加载表详情。"""
        path = self._table_cache_path(table_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return TableDetail(**data)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("缓存文件损坏: %s, error=%s", path, e)
            return None

    def _save_table_cache(self, table_id: int, detail: TableDetail) -> None:
        """将表详情保存到本地缓存。"""
        if not self.cache_dir:
            return
        path = self._table_cache_path(table_id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(detail.model_dump(), f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.warning("缓存写入失败: %s", e)
