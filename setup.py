# -*- coding: utf-8 -*-
import setuptools

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="QuantStudio",
    version="1.0.0",
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
        "numpy",# 矩阵运算
        "pandas",# 数据分析
        "scipy",# 科学计算
        "cvxpy",# 凸规划
        "matplotlib",# 绘图
        "statsmodels",# 概率统计
        "h5py",# HDF5 文件
        "cx-Oracle>=5.2.1",# Oracle 数据库
        "pymssql>=2.1.3",# SQL Server 数据库
        "mysql-connector-python<=8.0.16",# MySQL 数据库
        "pyodbc>=4.0.14",# ODBC
        "chardet>=3.0.4",# 解析字符编码
        "progressbar2>=3.10.1",# 进度条
        "fasteners>=0.14.0",# 进程文件锁
        "traits",# 对象参数
        "tushare",# tushare 数据源
        "xlrd",# 读 Excel 文件
        "xlwt",# 写 Excel 文件
        "openpyxl",# 读写 Excel 文件
        "tables",
        "Jinja2",
        "sympy",
        "ipython"
    ],
    package_data={"QuantStudio": ["Matlab/*", "Lib/*", "Resource/*"]}
)