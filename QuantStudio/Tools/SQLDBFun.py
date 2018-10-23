# coding=utf-8
"""SQL 数据库辅助函数"""

# 给定一个字符串list，用某个字符将其元素连接起来形成字符串,但每个字符串用到的元素数目不超过某个值
def genSQLInCondition(field_name, range_list, is_str=True, max_num=1000):
    nStr = len(range_list)
    if nStr<=max_num:
        if is_str:
            return field_name +" IN (\'"+"\', \'".join(range_list)+"\')"
        else:
            range_list = [str(iElement) for iElement in range_list]
            return field_name +" IN ("+', '.join(range_list)+")"
    else:
        if is_str:
            SQLStr = field_name+" IN (\'"+"\', \'".join(range_list[0:max_num])+"\')"
            i = max_num
            while i<nStr:
                SQLStr += " OR "+field_name+" IN (\'"+"\', \'".join(range_list[i:i+max_num])+"\')"
                i = i+max_num
        else:
            range_list = [str(iElement) for iElement in range_list]
            SQLStr = field_name+" IN ("+", ".join(range_list[0:max_num])+")"
            i = max_num
            while i<nStr:
                SQLStr += " OR "+field_name+" IN ("+", ".join(range_list[i:i+max_num])+")"
                i = i+max_num
        return SQLStr