import setuptools
import os

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="QuantStudio",
    version="0.0.4",
    author="scorpi000",
    author_email="scorpi000@sina.cn",
    maintainer="scorpi000",
    maintainer_email="scorpi000@sina.cn",
    description="Quant Studio",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://bitbucket.org/Scorpi000/quantstudio_v2/src/master/",
    license="GPLv3",
    platforms=["Windows"],
    python_requires=">=3.5",
    scripts=[],
    package_data={
    "QuantStudio": ["Matlab/*", "Lib/*", "Resource/*"]
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.14.4+mkl",
        "numexpr>=2.6.1",
        "pandas>=0.19.0",
        "scipy>=0.18.1",
        "matplotlib>=2.2.3",
        "seaborn>=0.7.1",
        "plotly>=2.0.6",
        "patsy",
        "statsmodels>=0.9.0",
        "h5py>=2.6.0",
        "cx-Oracle>=5.2.1",
        "pymongo>=3.4.0",
        "pymssql>=2.1.3",
        "pyodbc>=4.0.14",
        "arctic>=1.67.0",
        "chardet>=3.0.4",
        "progressbar2>=3.10.1",
        "pyface>=6.0.0",
        "traits>=4.6.0",
        "traitsui>=6.0.0",
        "bs4>=0.0.1",
        "tushare>=1.2.12",
        "Pyomo>=5.5.0",
        "xlrd>=0.9.0",
        "PyQt5>=5.9.2",
        "QScintilla>=2.10.2"
        # "pywin32>=220.1" if os.name=="nt"
    ]
)