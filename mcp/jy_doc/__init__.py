# -*- coding: utf-8 -*-
"""聚源数据库(Gildata)文档接入模块。

本模块随 QuantStudio 的 `mcp/jy_doc` MCP 服务一同分发，提供对聚源数据字典
平台(dd.gildata.com)的文档检索与查询能力：按数据库分类浏览、关键词搜索、
表详情与字段查看。数据通过 Gildata REST API 在线获取，并缓存到本地文件系统。

- `models`：数据模型（`TableDetail`、`ColumnInfo` 等）
- `fetcher`：Gildata API 抓取器（登录认证、目录树、表详情）
- `scraper`：本地搜索索引构建（展平目录树为扁平索引）
- `server`：MCP 服务入口（5 个工具）
"""
