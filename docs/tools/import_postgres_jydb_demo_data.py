# -*- coding: utf-8 -*-
import os
import re
import csv
import json
import time
import tempfile
import requests
import zipfile
import tarfile
import traceback
import datetime as dt
from typing import List, Dict, Any
from multiprocessing import Lock
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import psycopg2

import QuantStudio.api as QS

csv.field_size_limit(2048 * 2048)

# 产生一个有效的名字
def genAvailableName(header, all_names, check_header=True, ignore_case=False):
    if ignore_case:
        all_names = [iName.lower() for iName in all_names]
        LowerHeader = header.lower()
    else: LowerHeader = header
    if check_header and (LowerHeader not in all_names):
        return header
    i = 1
    CurName = header + str(i)
    while CurName in all_names:
        i += 1
        CurName = header + str(i)
    return CurName


class PostgresImporter:
    def __init__(self, host: str, port: str, database: str, username: str, password: str, import_dir: str = "exports", default_id_field: str = "JSID", conn_retry_num: int=3, conn_interval_seconds=30):
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.import_dir = import_dir
        self.checkpoint_file = "import_checkpoint.json"
        self._default_id_field = default_id_field
        self.conn_retry_num = conn_retry_num
        self.conn_inveral_seconds = conn_interval_seconds
        self.lock = Lock()
    
    @property
    def default_id_field(self):
        return self._default_id_field
    
    def get_connection(self):
        """获取数据库连接"""
        for _ in range(self.conn_retry_num):
            try:
                return psycopg2.connect(
                    host=self.host,
                    port=int(self.port), 
                    database=self.database,
                    user=self.username,
                    password=self.password,
                    keepalives=1,           # 开启 TCP keepalive
                    keepalives_idle=30,     # 30秒无数据开始发送探测包
                    keepalives_interval=10, # 探测包间隔
                    keepalives_count=5,     # 失败5次后断开
                    # # 连接超时设置
                    # connect_timeout=10,
                    # # 应用层心跳
                    # options='-c statement_timeout=0'
                )
            except Exception as e:
                pass
        raise Exception(f"数据库连接失败: {traceback.format_exc()}")

    def get_checkpoint(self, token: str, table_name: str) -> Dict[str, Any]:
        """获取断点信息"""
        checkpoint_file = self.import_dir + os.sep + table_name + os.sep + self.checkpoint_file
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                if checkpoint.get("token", None)==token:
                    return checkpoint
                else:
                    return {}
        return {}
    
    def save_checkpoint(self, table_name: str, checkpoint: Dict[str, Any]):
        """保存断点信息"""
        checkpoint_file = self.import_dir + os.sep + table_name + os.sep + self.checkpoint_file
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    
    def check_table_exists(self, table_name: str):
        conn = self.get_connection()
        cursor = conn.cursor()
        # 检查表是否存在
        try:
            cursor.execute(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = '{table_name.lower()}'
                )
            """)
            
            exists = cursor.fetchone()[0]
        finally:
            cursor.close()
            conn.close()
        return exists

    def create_table_if_not_exists(self, table_name: str, columns_info: List[Dict[str, Any]]):
        """如果表不存在则创建表"""
        with self.lock:
            # 检查表是否存在
            if self.check_table_exists(table_name): return True

            conn = self.get_connection()
            cursor = conn.cursor()
            try:
                # 构建建表语句
                column_defs, pk_list = [], []
                for col in columns_info:
                    pg_type = self.map_sqlserver_type_to_postgres(col['type'])
                    nullable = 'NOT NULL' if col['nullable']=="False" else 'NULL'
                    column_defs.append(f'{col["name"]} {pg_type} {nullable}')
                    if col["pk"] == "YES": pk_list.append(col["name"])
                if pk_list:
                    pk_def = f" PRIMARY KEY ({','.join(pk_list)})"
                else:
                    pk_def = ""
                create_sql = f'CREATE TABLE IF NOT EXISTS {table_name} ({", ".join(column_defs)}{pk_def})'
                cursor.execute(create_sql)
                conn.commit()
                print(f"表 {table_name} 已创建")
            finally:
                cursor.close()
                conn.close()
            return False
    
    def create_index(self, table_name: str, index_info):
        """创建索引"""
        db_index_info = self.get_index_info(table_name=table_name)
        db_index_list = [index_name.lower() for index_name in db_index_info["index_name"].tolist()]
        conn = self.get_connection()
        cursor = conn.cursor()
        for index_name in index_info["index_name"].unique():
            if index_name.lower() in db_index_list:
                print(f"表 {table_name} 的索引 {index_name} 已存在，跳过创建")
                continue
            if index_name.lower() == table_name.lower():
                new_index_name = genAvailableName(index_name, all_names=[table_name]+index_info["index_name"].tolist(), check_header=False, ignore_case=True)
                print(f"表名 {table_name} 和索引名相同，修改索引名为: {new_index_name}")
            else:
                new_index_name = index_name
            if new_index_name.lower() in db_index_list:
                print(f"表 {table_name} 的索引 {new_index_name} 已存在，跳过创建")
                continue
            try:
                iinfo = index_info[index_info["index_name"]==index_name]
                constraint_type = iinfo["constraint_type"].unique()
                if constraint_type.shape[0] > 1:
                    print(f"表 {table_name} 的索引 {index_name} 有多种 constraint_type: {constraint_type}")
                constraint_type = constraint_type[0]
                if constraint_type.lower() in ("unique index", "unique constraint"):
                    unique = " UNIQUE "
                else:
                    unique = " "
                col_def = ", ".join(iinfo['column_name'].iloc[i] + " " + iinfo['sort_direction'].iloc[i] for i in range(iinfo.shape[0]))
                sql = f"""CREATE{unique}INDEX {new_index_name} ON {table_name}({col_def})"""
                cursor.execute(sql)
                conn.commit()
                print(f"表 {table_name} 的索引 {new_index_name} 创建完成")
            except:
                print(f"表 {table_name} 的索引 {new_index_name} 创建失败: {traceback.format_exc()}")
        cursor.close()
        conn.close()

    def get_unique_fields(self, index_info):
        unique_index_info = index_info[index_info["constraint_type"].str.lower().isin(("unique index", "unique constraint"))]
        index_name_list = set(unique_index_info["index_name"].tolist())
        ID_JSID_index = set(unique_index_info[unique_index_info["column_name"].isin(("ID", "JSID"))]["index_name"].tolist())
        selected_index = index_name_list.difference(ID_JSID_index)
        if selected_index:
            selected_index = sorted(selected_index)[0]
            return unique_index_info[unique_index_info["index_name"]==selected_index]["column_name"].tolist()
        else:
            return None

    def map_sqlserver_type_to_postgres(self, sqlserver_type: str) -> str:
        """SQL Server 类型映射到 PostgreSQL 类型"""
        type_mapping = {
            'int': 'INTEGER',
            'bigint': 'BIGINT',
            'smallint': 'SMALLINT',
            'tinyint': 'SMALLINT',
            'bit': 'BOOLEAN',
            'decimal': 'DECIMAL',
            'numeric': 'NUMERIC',
            'float': 'DOUBLE PRECISION',
            'real': 'REAL',
            'money': 'DECIMAL(19,4)',
            'smallmoney': 'DECIMAL(10,4)',
            'char': 'CHAR',
            'varchar': 'VARCHAR',
            'text': 'TEXT',
            'nchar': 'CHAR',
            'nvarchar': 'VARCHAR',
            'ntext': 'TEXT',
            'datetime': 'TIMESTAMP',
            'datetime2': 'TIMESTAMP',
            'smalldatetime': 'TIMESTAMP',
            'date': 'DATE',
            'time': 'TIME',
            'datetimeoffset': 'TIMESTAMPTZ',
            'uniqueidentifier': 'UUID',
            'binary': 'BYTEA',
            'varbinary': 'BYTEA',
            'image': 'BYTEA',
            'xml': 'XML'
        }
        
        # 特殊映射
        if sqlserver_type.lower()=="varchar(-1)": return "TEXT"

        # 处理带长度的类型
        import re
        match = re.match(r'(\w+)(?:\((\d+)(?:,(\d+))?\))?', sqlserver_type.lower())
        if match:
            base_type = match.group(1)
            if base_type in type_mapping:
                pg_type = type_mapping[base_type]
                if match.group(2):
                    if match.group(3):  # 有精度和小数位数
                        pg_type += f"({match.group(2)},{match.group(3)})"
                    else:  # 只有长度
                        pg_type += f"({match.group(2)})"
                return pg_type
        
        return 'TEXT'  # 默认类型
    
    def create_del_table_if_not_exists(self, table_name:str="JYDB_DeleteRec"):
        # 检查表是否存在
        with self.lock:
            if self.check_table_exists(table_name): return True
            conn = self.get_connection()
            cursor = conn.cursor()
            try:
                create_sql = f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                    TABLENAME varchar(100) NOT NULL,
                    RECID bigint NOT NULL,
                    XGRQ timestamp(6) NOT NULL,
                    ID bigint NOT NULL,
                    JSID bigint NOT NULL
                    )
                """
                cursor.execute(create_sql)
                conn.commit()

                # 创建索引
                sql = f"""CREATE INDEX IX_JYDB_DeleteRec_DATE ON {table_name}(xgrq ASC)"""
                cursor.execute(sql)
                sql = f"""CREATE INDEX IX_JYDB_DeleteRec_ID ON {table_name}(id ASC)"""
                cursor.execute(sql)
                sql = f"""CREATE INDEX IX_JYDB_DeleteRec_JSID ON {table_name}(jsid ASC)"""
                cursor.execute(sql)
                sql = f"""CREATE UNIQUE INDEX IX_JYDB_DeleteRec_TNJS ON {table_name}(tablename ASC, jsid ASC)"""
                cursor.execute(sql)
                conn.commit()

                print(f"删除表 {table_name} 已创建")
            finally:
                cursor.close()
                conn.close()
            return False

    # 删除表，同时清空 del_table 中的删除记录
    def delete_table(self, table_name:str, del_table_name:str="JYDB_DeleteRec"):
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            sql = f"""DROP TABLE IF EXISTS {table_name}"""
            cursor.execute(sql)
            sql = f"""DELETE FROM {del_table_name} WHERE TABLENAME='{table_name}'"""
            conn.commit()
            print(f"删除表: {table_name}")
        finally:
            cursor.close()
            conn.close()

    def get_max_id(self, table_name:str, id_field:str="JSID"):
        exists = self.check_table_exists(table_name)
        if not exists: return None
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(f"""SELECT MAX({id_field}) FROM {table_name}""")
            max_id = cursor.fetchone()
        finally:
            cursor.close()
            conn.close()
        if max_id: return max_id[0]
        else: return None
    
    def get_del_max_id(self, table_name:str, del_table_name:str="JYDB_DeleteRec", id_field:str="JSID"):
        exists = self.check_table_exists(table_name)
        if not exists: return None
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(f"""SELECT MAX({id_field}) FROM {del_table_name} WHERE TABLENAME = '{table_name}'""")
            max_id = cursor.fetchone()
        finally:
            cursor.close()
            conn.close()
        if max_id: return max_id[0]
        else: return None

    def get_index_info(self, table_name: str):
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            sql = f"""
                SELECT 
                    indexname AS index_name,
                    indexdef AS index_def
                FROM pg_indexes
                WHERE tablename = '{table_name.lower()}'
                AND schemaname = 'public'
            """
            cursor.execute(sql)
            index_info = cursor.fetchall()
            return pd.DataFrame(index_info, columns=[col.name for col in cursor.description])
        finally:
            cursor.close()
            conn.close()

    def import_del_table(self, token:str, table_name:str, exec_del:bool=True, del_table_name:str="JYDB_DeleteRec", cursor=None, conn=None):
        del_file = self.import_dir + os.sep + table_name + os.sep + f"{token}-DEL.csv"
        if not os.path.isfile(del_file):
            print(f"表 {table_name} 无删除文件!")
            return
        db_table_name = table_name.split("-")[0]
        insert_sql = f'INSERT INTO {del_table_name} (TABLENAME, RECID, XGRQ, ID, JSID) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (JSID, TABLENAME) DO UPDATE SET RECID=EXCLUDED.RECID, XGRQ=EXCLUDED.XGRQ, ID=EXCLUDED.ID'
        with open(del_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, escapechar="\\")
            batch = list(reader)
            if batch and (batch[-1][0]=="salt"): batch = batch[:-1]# 去掉最后一行的 salt
        imported_cursor = (cursor is not None)
        if not imported_cursor:
            conn = self.get_connection()
            cursor = conn.cursor()
        try:
            # 先执行删除表数据插入
            cursor.executemany(insert_sql, batch)
            # 再删除目标表中的数据
            if exec_del:
                del_sql = f"""DELETE FROM {db_table_name} WHERE ID IN ('{"','".join(row[1] for row in batch)}')"""
                cursor.execute(del_sql)
            conn.commit()
        finally:
            if not imported_cursor:
                cursor.close()
                conn.close()
        print(f"表 {table_name} 的删除文件导入完成!")

    def clear_redundant_data(self, db_table_name: str, first_idx, last_idx=None, id_field:str="JSID", cursor=None, conn=None):
        imported_cursor = (cursor is not None)
        if not imported_cursor:
            conn = self.get_connection()
            cursor = conn.cursor()
        try:
            sql = f"""DELETE FROM {db_table_name} WHERE {id_field} >= {first_idx}"""
            if last_idx: sql += f""" AND {id_field} <= {last_idx}"""
            cursor.execute(sql)
            conn.commit()
        finally:
            if not imported_cursor:
                cursor.close()
                conn.close()

    def handle_unique_insert(self, table_name, ifile, batch, e, insert_sql, columns_info):
        print(f"表 {table_name} 文件 {ifile} 出现唯一性错误: {e}, 开始处理...")
        # 识别出错的行
        pattern = r'\(([^)]+)\)'
        matches = re.findall(pattern, e.diag.message_detail)
        if len(matches)!=2:
            raise Exception(f"表 {table_name} 文件 {ifile} 无法识别插入唯一性出错的行，错误信息: {e}")
        else:
            cols, row = matches
        cols = [col.strip().lower() for col in cols.split(",")]
        row = [val.strip() for val in row.split(",")]
        mask = None
        for i, col in enumerate(cols):
            col_info = columns_info[col]
            val = row[i]
            if "int" in col_info["type"].lower(): val = int(val)
            if mask is None:
                mask = (batch[:, col_info["idx"]]==val)
            else:
                mask = mask & (batch[:, col_info["idx"]]==val)
        err_rows, batch = batch[mask].tolist(), batch[~mask]
        if batch.shape[0]==0: return err_rows
        if not err_rows:
            raise Exception(f"表 {table_name} 文件 {ifile} 未过滤出插入唯一性出错的行, 错误信息: {e}")
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.executemany(insert_sql, batch.tolist())
            conn.commit()
        except psycopg2.errors.UniqueViolation as e:# 处理唯一性错误
            cursor.close()
            conn.close()
            err_rows += self.handle_unique_insert(table_name=table_name, ifile=ifile, batch=batch, e=e, insert_sql=insert_sql, columns_info=columns_info)
        except Exception as e:
            raise e
        finally:
            cursor.close()
            conn.close()
        return err_rows

    def import_table(self, token: str, table_name: str, resume: bool = True, id_field="JSID", del_table_name:str="JYDB_DeleteRec") -> int:
        print(f"开始导入任务 {token} 的表 {table_name}...")
        db_table_name = table_name.split("-")[0]
        header_file =  self.import_dir + os.sep + table_name + os.sep + f"{token}.csv"
        if not os.path.exists(header_file):
            raise FileNotFoundError(f"表头文件不存在: {header_file}")
        with open(header_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, escapechar="\\")
            headers = next(reader)# 读取表头
            data_type = next(reader)# 数据类型
            nullable = next(reader)# 是否可空
            pk = next(reader)# 是否为主键
        # 创建表结构信息
        columns_info = [{'name': col, 'type': data_type[i], 'nullable': nullable[i], 'pk': pk[i]} for i, col in enumerate(headers)]
        # 创建表（如果不存在）
        table_exists = self.create_table_if_not_exists(db_table_name, columns_info)
        self.create_del_table_if_not_exists()
        # 创建索引
        index_file = self.import_dir + os.sep + table_name + os.sep + f"{token}-INDEX.csv"
        if os.path.isfile(index_file):
            with open(index_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, escapechar="\\")
                index_info = list(reader)
                if index_info and (index_info[-1][0]=="salt"): index_info = index_info[:-1]# 去掉最后一行的 salt
            index_info = pd.DataFrame(index_info[1:], columns=index_info[0])
            self.create_index(db_table_name, index_info)
            unique_fields = self.get_unique_fields(index_info=index_info)
            unique_fields = [ifield.lower() for ifield in unique_fields]
        else:
            unique_fields = None
        if not unique_fields:
            print(f"表 {db_table_name} 无用于判断行唯一性的字段" )
        else:
            print(f"表 {db_table_name} 将根据字段 {unique_fields} 判断行的唯一性" )
        
        # 获取断点信息
        checkpoint = self.get_checkpoint(token, table_name) if resume else {}
        imported_rows = checkpoint.get('imported_rows', 0)
        idx = int(checkpoint.get("idx", 0))
        
        # 读取CSV文件列表，并清理表里多余的数据
        file_list = sorted(ifile for ifile in os.listdir(self.import_dir + os.sep + table_name) if ifile.startswith(token+"-") and ifile.endswith(".csv") and (ifile.split(".")[0].split("-")[-1].lower() not in ("del", "index", "unique_error")) and int(ifile.split(".")[0].split("-")[-1]) >= idx)
        # if table_exists and file_list:
        #     id_col_idx = [i for i, col in enumerate(columns_info) if col["name"].lower()==id_field.lower()][0]
        #     with open(os.path.join(self.import_dir, table_name, file_list[0]), 'r', encoding='utf-8') as f:
        #         first_row = next(csv.reader(f, escapechar="\\"))
        #     first_idx = first_row[id_col_idx]
        #     with open(os.path.join(self.import_dir, table_name, file_list[-1]), 'r', encoding='utf-8') as f:
        #         last_row = next(iter(deque(csv.reader(f, escapechar="\\"), maxlen=1)))
        #     last_idx = last_row[id_col_idx]
        #     self.clear_redundant_data(db_table_name=db_table_name, first_idx=first_idx, last_idx=last_idx, id_field=id_field)

        # 读取删除表数据
        del_file = self.import_dir + os.sep + table_name + os.sep + f"{token}-DEL.csv"
        if os.path.isfile(del_file):
            with open(del_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, escapechar="\\")
                del_data = list(reader)
                if del_data and (del_data[-1][0]=="salt"): del_data = del_data[:-1]# 去掉最后一行的 salt
            del_data = pd.DataFrame(del_data, columns=["TABLENAME", "RECID", "XGRQ", "ID", "JSID"])
            del_ids = del_data["RECID"].astype(int).tolist()
        else:
            del_ids = None
        
        # 导入删除表数据并执行删除
        self.import_del_table(token=token, table_name=table_name, exec_del=table_exists, del_table_name=del_table_name)
        
        # 构建插入语句
        columns_str = ', '.join(headers)
        placeholders = ', '.join(['%s'] * len(headers))
        update_str = ", ".join(f"{col}=EXCLUDED.{col}" for col in headers if col.lower() != "id")
        insert_sql = f'INSERT INTO {db_table_name} ({columns_str}) VALUES ({placeholders}) ON CONFLICT (id) DO UPDATE SET {update_str}'
        
        id_col_idx, datetime_col_idx, float_col_idx, int_col_idx, text_col_idx = None, [], [], [], []
        for i, col in enumerate(columns_info):
            if col["name"].lower()=="id": id_col_idx = i
            if col["type"].lower().startswith("date"): datetime_col_idx.append(i)
            elif col["type"].lower().startswith("float"): float_col_idx.append(i)
            elif "int" in col["type"].lower(): int_col_idx.append(i)
            elif col["type"].lower().startswith("text") or col["type"].lower()=="varchar(-1)": text_col_idx.append(i)
        if id_col_idx is None: raise Exception(f"没有找到 ID 字段: {columns_info}")
        
        total_imported = imported_rows
        conn = self.get_connection()
        cursor = conn.cursor()
        # 导入表数据
        for ifile in file_list:
            try:
                with open(self.import_dir + os.sep + table_name + os.sep + ifile, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f, escapechar="\\")
                    batch = list(reader)
                # with open(self.import_dir + os.sep + table_name + os.sep + ifile, 'r', encoding='utf-8') as f:
                #     row_num = sum(1 for _ in f)
                # if row_num != len(batch):# 发生了换行识别错误
                #     def parse_csv_line(line):
                #         if line.endswith("\\\n"):
                #             reader = csv.reader(StringIO(line[:-2]), escapechar="\\")
                #         else:
                #             reader = csv.reader(StringIO(line), escapechar="\\")
                #         return next(reader)
                #     with open(self.import_dir + os.sep + table_name + os.sep + ifile, 'r', encoding='utf-8') as f:
                #         batch = [parse_csv_line(line) for line in f]
                if batch and (batch[-1][0]=="salt"): batch = batch[:-1]# 去掉最后一行的 salt

                batch = np.array(batch, dtype="O")
                batch = np.where(batch!="", batch, None)
                # 处理特殊类型的字段
                # if datetime_col_idx:
                #     for i in datetime_col_idx:
                #         batch[:, i] = np.where(batch[:, i]!="", batch[:, i], None)
                if float_col_idx:
                    for i in float_col_idx:
                        batch[:, i] = batch[:, i].astype(float)
                if int_col_idx:
                    for i in int_col_idx:
                        mask = pd.isnull(batch[:, i])
                        batch[:, i] = np.where(mask, batch[:, i], np.where(mask, 0, batch[:, i]).astype(int))
                if text_col_idx:
                    modify_func = lambda x: x.replace("\x00", "") if pd.notnull(x) else None
                    for i in text_col_idx:
                        batch[:, i] = [modify_func(x) for x in batch[:, i]]
                # 处理被删除的行
                if del_ids is not None:
                    batch = batch[~np.isin(batch[:, id_col_idx], del_ids)]
                
                cursor.executemany(insert_sql, batch.tolist())
                conn.commit()
            except psycopg2.errors.UniqueViolation as e:# 处理唯一性错误
                cursor.close()
                conn.close()
                try:
                    unique_err_rows = self.handle_unique_insert(table_name=table_name, ifile=ifile, batch=batch, e=e, insert_sql=insert_sql, columns_info={col["name"].lower(): col | {"idx": i} for i, col in enumerate(columns_info)})
                except:
                    print(f"表 {table_name} 文件 {ifile} 唯一性错误处理失败, 导入失败!")
                    raise e
                conn = self.get_connection()
                cursor = conn.cursor()
                if unique_err_rows:
                    unique_err_file = ".".join(ifile.split(".")[:-1])+"-UNIQUE_ERROR.csv"
                    with open(self.import_dir + os.sep + table_name + os.sep + unique_err_file, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f, escapechar="\\")
                        writer.writerows(unique_err_rows)
                    unique_err_update_str = ", ".join(f"{col}=EXCLUDED.{col}" for col in headers if col.lower() not in unique_fields)
                    unique_err_insert_sql = f'INSERT INTO {db_table_name} ({columns_str}) VALUES ({placeholders}) ON CONFLICT ({",".join(unique_fields)}) DO UPDATE SET {unique_err_update_str}'
                    try:
                        cursor.executemany(unique_err_insert_sql, unique_err_rows)
                        conn.commit()
                    except Exception as e:
                        cursor.close()
                        conn.close()
                        raise e
                    else:
                        print(f"表 {table_name} 文件 {ifile} 唯一性错误处理完成, 通过调整唯一键写入成功")
            except Exception as e:
                cursor.close()
                conn.close()
                print(f"表 {table_name} 文件 {ifile} 导入失败!")
                raise e
            # 保存断点
            total_imported += batch.shape[0]
            idx += 1
            print(f"表 {table_name} 文件 {ifile} 导入完成!")
            if resume:
                self.save_checkpoint(table_name, {
                    'token': token,
                    'imported_rows': total_imported,
                    'import_time': dt.datetime.now().isoformat(),
                    'idx': idx,
                })

        cursor.close()
        conn.close()
        return total_imported


def download_and_extract(url):
    """
    下载压缩包到系统临时目录并解压缩
    
    Args:
        url (str): 压缩包的链接地址
    
    Returns:
        str: 解压缩后的目录地址
    """
    # 创建临时文件路径
    temp_dir = tempfile.mkdtemp()
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename:
        filename = "downloaded_archive"
    
    # 确保文件扩展名存在
    if not any(filename.endswith(ext) for ext in ['.zip', '.tar.gz', '.tgz', '.tar']):
        filename += '.zip'  # 默认添加.zip扩展名
    
    file_path = os.path.join(temp_dir, filename)
    
    # 下载文件
    print(f"正在下载: {url}")
    response = requests.get(url)
    response.raise_for_status()  # 检查请求是否成功
    
    with open(file_path, 'wb') as f:
        f.write(response.content)
    
    print(f"下载完成: {file_path}")
    
    # 创建解压目录
    extract_dir = os.path.join(temp_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    
    # 根据文件类型解压缩
    if filename.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif filename.endswith(('.tar.gz', '.tgz')) or '.tar.gz' in filename:
        with tarfile.open(file_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_dir)
    elif filename.endswith('.tar'):
        with tarfile.open(file_path, 'r') as tar_ref:
            tar_ref.extractall(extract_dir)
    else:
        raise ValueError(f"不支持的文件格式: {filename}")
    
    print(f"解压完成: {extract_dir}")
    return extract_dir

def main(url:str="https://gitee.com/scorpi000/QSImage/raw/master/DemoData/JYDB.zip"):
    print("下载 JYDB Demo 数据...")
    ExtractedPath = download_and_extract(url)

    JYDB = QS.Factor.JYDB().connect()

    config = {
        'host': JYDB.Args.IPAddr,
        'port': str(JYDB.Args.Port),
        'database': JYDB.Args.DBName,
        'username': JYDB.Args.User,
        'password': JYDB.Args.Pwd,
        'import_dir': ExtractedPath
    }
    importer = PostgresImporter(**config)

    for itable in os.listdir(importer.import_dir):
        imported_rows = importer.import_table(token="aha", table_name=itable, del_table_name="JYDB_DeleteRec", resume=False)
        print(f"{itable} 导入 {imported_rows} 行")


if __name__=="__main__":
    main()