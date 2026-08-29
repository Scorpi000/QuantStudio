"""比较两个 HDF5DB 因子库中给定表的数据一致性。

功能
----
给定两个因子库的主目录 (MainDir) 以及需要比较的表名列表, 脚本会:

1. 分别连接两个因子库, 遍历指定表列表;
2. 对每张表, 提取其下所有因子的数据 (通过 readFactorData 读取为 DataFrame);
3. 逐因子比较两个库中同名因子的 DataFrame 是否完全一致;
4. 输出每个因子的比较结果 (一致 / 差异详情), 并最终给出汇总。

差异判定
--------
两个 DataFrame 被视为一致的条件为:
    * model 形状 (行数/列数) 相同;
    * 行索引 (DateTime) 与列索引 (ID) 完全相同;
    * 对应位置的值完全相等 (NaN 与 NaN 视为相等)。

任何一项不满足即判定为不一致, 并输出具体差异信息。

使用方法
--------
    # 比较指定的表列表
    python scripts/compare_hdf5db.py \
        --dir1 <因子库1主目录> \
        --dir2 <因子库2主目录> \
        --tables <表名1> <表名2> ...

    # 比较第一个因子库下的所有表
    python scripts/compare_hdf5db.py \
        --dir1 <因子库1主目录> \
        --dir2 <因子库2主目录> \
        --all-tables-of-dir1

示例
----
    python scripts/compare_hdf5db.py \
        --dir1 C:/data/fdb_a \
        --dir2 C:/data/fdb_b \
        --tables stock_cn_day_bar stock_cn_factor_value

依赖
----
    QuantStudio.Factor.api.HDF5DB, pandas, numpy
"""

import argparse

import pandas as pd

from QuantStudio.Factor.api import HDF5DB


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame):
    """比较两个 DataFrame 是否一致, 不一致时返回差异描述。

    Args:
        df1: 第一个 DataFrame。
        df2: 第二个 DataFrame。

    Returns:
        (是否一致, 差异描述字符串)。一致时差异描述为空字符串。
    """
    if df1.shape != df2.shape:
        return False, f"形状不一致: {df1.shape} vs {df2.shape}"

    if not df1.index.equals(df2.index):
        return False, "行索引 (DateTime) 不一致"

    if not df1.columns.equals(df2.columns):
        return False, "列索引 (ID) 不一致"

    try:
        # NaN 与 NaN 视为相等
        equal_mask = (df1 == df2) | (df1.isna() & df2.isna())
    except Exception as e:
        return False, f"逐值比较失败: {e}"

    if not equal_mask.all().all():
        n_diff = int((~equal_mask).sum().sum())
        return False, f"存在 {n_diff} 个位置的值不一致"

    return True, ""


def main():
    parser = argparse.ArgumentParser(
        description="比较两个 HDF5DB 因子库中给定表的数据一致性"
    )
    parser.add_argument(
        "--dir1", required=True, help="第一个因子库的主目录 (MainDir)"
    )
    parser.add_argument(
        "--dir2", required=True, help="第二个因子库的主目录 (MainDir)"
    )
    parser.add_argument(
        "--tables", nargs="+", help="需要比较的表名列表; 与 --all-uses-dir1 互斥"
    )
    parser.add_argument(
        "--all-tables-of-dir1",
        action="store_true",
        help="使用第一个因子库 (dir1) 下的所有表作为待比较表列表",
    )
    parser.add_argument(
        "--ignore-ids",
        action="store_true",
        help="忽略 ID 不一致, 只比较两库 ID 交集部分是否一致",
    )
    parser.add_argument(
        "--ignore-datetimes",
        action="store_true",
        help="忽略时间序列 (DateTime) 不一致, 只比较两库时点交集部分是否一致",
    )
    args = parser.parse_args()

    fdb1 = HDF5DB(args={"MainDir": args.dir1}).connect()
    fdb2 = HDF5DB(args={"MainDir": args.dir2}).connect()

    if args.all_tables_of_dir1:
        args.tables = fdb1.TableNames
    elif not args.tables:
        parser.error("必须提供 --tables 列表, 或使用 --all-tables-of-dir1")

    total_consistent = 0
    total_inconsistent = 0

    for table_name in args.tables:
        for fdb, dir_path, tag in ((fdb1, args.dir1, "库1"), (fdb2, args.dir2, "库2")):
            if table_name not in fdb.TableNames:
                print(f"[跳过] 表 '{table_name}' 不存在于 {tag} ({dir_path})")
                continue

        print(f"\n=== 表: {table_name} ===")

        if table_name not in fdb1.TableNames or table_name not in fdb2.TableNames:
            continue

        table1 = fdb1.getTable(table_name)
        table2 = fdb2.getTable(table_name)

        factors1 = set(table1.FactorNames)
        factors2 = set(table2.FactorNames)

        only_in_1 = factors1 - factors2
        only_in_2 = factors2 - factors1
        for factor_name in sorted(only_in_1):
            print(f"[差异] 因子 '{factor_name}' 仅存在于库1")
            total_inconsistent += 1
        for factor_name in sorted(only_in_2):
            print(f"[差异] 因子 '{factor_name}' 仅存在于库2")
            total_inconsistent += 1

        common_factors = sorted(factors1 & factors2)
        for factor_name in common_factors:
            ids1 = table1.getID(ifactor_name=factor_name)
            dts1 = table1.getDateTime(ifactor_name=factor_name)
            ids2 = table2.getID(ifactor_name=factor_name)
            dts2 = table2.getDateTime(ifactor_name=factor_name)

            # 根据参数决定 ID 与时点索引的取值方式;
            # 不忽略时取两库并集, 忽略时取交集.
            if args.ignore_ids:
                ids = sorted(set(ids1) & set(ids2))
            else:
                ids = sorted(set(ids1) | set(ids2))
            if args.ignore_datetimes:
                dts = sorted(set(dts1) & set(dts2))
            else:
                dts = sorted(set(dts1) | set(dts2))

            df1 = table1.readFactorData(factor_name, ids=ids, dts=dts)
            df2 = table2.readFactorData(factor_name, ids=ids, dts=dts)

            consistent, diff_msg = compare_dataframes(df1, df2)
            if consistent:
                total_consistent += 1
                print(f"[一致] {factor_name}")
            else:
                total_inconsistent += 1
                print(f"[差异] {factor_name}: {diff_msg}")

    print("\n=== 汇总 ===")
    print(f"一致因子数: {total_consistent}")
    print(f"不一致因子数: {total_inconsistent}")

    if total_inconsistent:
        print("结论: 两个因子库在给定表范围内存在不一致。")
        return 1
    print("结论: 两个因子库在给定表范围内完全一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
