export PGCLIENTENCODING=UTF8

# 先导出表结构 -h 主机名 -U 用户名 -d 数据库名 -t 表名
pg_dump -h localhost -U postgres -d JYDB -t secumain --schema-only > secumain.sql

# 再导出部分数据（使用 COPY 命令）
psql -h localhost -U postgres -d JYDB -c "COPY (SELECT * FROM secumain WHERE innercode IN (3, 11, 310976, 1679, 300284, 398635)) TO STDOUT" >> secumain.sql