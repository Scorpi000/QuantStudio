"""聚源元数据路径兼容测试，不连接数据库。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from QuantStudio.Factor.JYDB import _importInfo


class JYDBInfoPathTests(unittest.TestCase):
    """Pydantic 的 FilePath 必须可用于元数据加载。"""

    def test_path_resource_loads_json_without_cache_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"metadata.json"
            path.write_text(json.dumps({"TableInfo":{"行情":{"DBTableName":"prices"}},
                "FactorInfo":{"行情":{"收盘价":{"DBFieldName":"close"}}},
                "ExchangeInfo":{"83":{"Exchange":"SSE"}},
                "SecurityInfo":{"8":{"SecurityType":"公募基金"}}},ensure_ascii=False),encoding="utf-8")
            tables,fields,exchanges,securities = _importInfo(None,path,Mock(),out_info=True)
            self.assertEqual(tables.loc["行情","DBTableName"],"prices")
            self.assertEqual(fields.loc[("行情","收盘价"),"DBFieldName"],"close")
            self.assertEqual(exchanges.loc["83","Exchange"],"SSE")


if __name__ == "__main__":
    unittest.main()
