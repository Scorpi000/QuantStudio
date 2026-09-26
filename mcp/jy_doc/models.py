# -*- coding: utf-8 -*-
"""聚源数据库文档数据模型定义。

定义了表信息、字段信息、数据库分类、搜索结果等核心数据结构，使用 Pydantic v2 进行数据校验。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ColumnInfo(BaseModel):
    """数据库表的字段信息。

    Attributes:
        name: 字段名称（英文）
        chinese_name: 字段中文名
        data_type: 数据类型（如 varchar、int、decimal）
        is_nullable: 是否允许为空
        remark: 字段说明
    """

    name: str
    chinese_name: str = ""
    data_type: str = ""
    is_nullable: bool = True
    remark: str = ""


class TableDetail(BaseModel):
    """数据库表的完整详情。

    Attributes:
        table_id: Gildata 平台中的表 ID
        table_name: 表的中文名
        base_table_name: 数据库中的物理表名
        path: 文档路径（层级路径）
        description: 表说明
        update_frequency: 数据更新频率
        columns: 字段列表
        slave_columns: 从表字段列表（原始 JSON）
        unique_index: 唯一索引信息（原始 JSON）
        created_date: 创建日期
        last_modified_date: 最后修改日期
    """

    table_id: int
    table_name: str
    base_table_name: str
    path: str = ""
    description: str = ""
    update_frequency: str = ""
    columns: list[ColumnInfo] = Field(default_factory=list)
    slave_columns: list[dict] = Field(default_factory=list)
    unique_index: dict = Field(default_factory=dict)
    created_date: str = ""
    last_modified_date: str = ""


class DatabaseInfo(BaseModel):
    """聚源数据库库级别信息。

    Attributes:
        id: 库 ID（Gildata 平台中的 product group ID）
        name: 库名，如 "国内上市公司数据库"
        description: 库说明
    """

    id: int
    name: str
    description: str = ""


class SearchResultItem(BaseModel):
    """搜索结果中的一条记录。

    Attributes:
        table_id: Gildata 平台中的表 ID
        table_name: 表的中文名
        base_table_name: 数据库中的物理表名
        path: 文档路径
        description: 表说明
        category: 所属数据库分类
        score: 匹配分数
    """

    table_id: int
    table_name: str
    base_table_name: str
    path: str = ""
    description: str = ""
    category: str = ""
    score: float = 0.0
