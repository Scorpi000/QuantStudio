# Notebook 使用方法

## nbconvert 直接执行

```bash
# 原地执行，将输出写回原文件
jupyter nbconvert --to notebook --execute --inplace scripts/compare_hdf5db.ipynb

# 执行并输出到另一个文件
python -m nbconvert --to notebook --execute scripts/compare_hdf5db.ipynb --output compared.ipynb
```

说明：`--execute` 会顺序执行所有代码 cell；`--inplace` 表示结果写回原文件，否则生成新文件。执行耗时视数据量而定，大库可加 `--ExecutePreprocessor.timeout=600` 来放宽超时（见 nbconvert 文档 `ExecutePreprocessor`）。

## papermill 参数化运行（可传入参数）

本 notebook 第一个 cell 标记了 `parameters`，papermill 可据此注入新参数值，
无需手动改 cell：

```bash
papermill scripts/compare_hdf5db.ipynb out.ipynb \
    -p dir1 "D:\\Data\\HDF5DB" \
    -p dir2 "D:\\Data\\HDF5DB_Test" \
    -p target_table "stock_cn_status" \
    -p target_factor "if_listed"
```

说明：
- `-p 参数名 值` 会覆盖 parameters cell 中同名变量的取值；
- Windows 下反斜杠路径需写成 `\\`（JSON 转义）或改用正斜杠；
- papermill 会输出一个新的 notebook（此处为 `out.ipynb`），原文件不变。