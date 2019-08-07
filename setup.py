# -*- coding: utf-8 -*-
import setuptools
import os

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="QuantStudio",
    version="0.0.9",
    author="scorpi000",
    author_email="scorpi000@sina.cn",
    maintainer="scorpi000",
    maintainer_email="scorpi000@sina.cn",
    description="Quant Studio",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/Scorpi000/QuantStudio/",
    license="GPLv3",
    platforms=["Windows"],
    python_requires=">=3.5",
    scripts=[],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        'Topic :: Office/Business :: Financial :: Investment'
    ],
    install_requires=[
        "numpy>=1.14.4+mkl",# 矩阵运算
        "numexpr>=2.6.1",# 加速数值计算
        "pandas>=0.19.0,<0.25.0",# 数据分析
        "scipy>=0.18.1",# 科学计算
        "cvxpy>=1.0.0",# 凸规划
        "matplotlib>=2.2.3",# 绘图
        "seaborn>=0.9.0",# 绘图
        "plotly>=2.0.6",# 绘图
        "patsy",# 描述统计模型
        "statsmodels>=0.9.0",# 概率统计
        "h5py>=2.6.0",# HDF5 文件
        "cx-Oracle>=5.2.1",# Oracle 数据库
        "pymssql>=2.1.3",# SQL Server 数据库
        "mysql-connector-python",# MySQL 数据库
        "pyodbc>=4.0.14",# ODBC
        "pymongo>=3.4.0",# mongodb
        "arctic>=1.67.0",# Arctic 时间序列数据库
        "chardet>=3.0.4",# 解析字符编码
        "progressbar2>=3.10.1",# 进度条
        "fasteners>=0.14.0",# 进程文件锁
        "pyface>=6.0.0",# 对象参数
        "traits>=4.6.0",# 对象参数
        "traitsui>=6.0.0",# 对象参数
        "bs4>=0.0.1",# 解析网页
        "tushare>=1.2.12",# tushare 数据源
        "xlrd>=0.9.0",# 读 Excel 文件
        "xlwt>=1.3.0",# 写 Excel 文件
        "PyQt5>=5.9.2",# GUI
    ],
    package_data={"QuantStudio": ["Matlab/*", "Lib/*", "Resource/*"]}
)