"""BaoStockDB 本地配置回归，不访问远程服务。"""
from pathlib import Path
from shutil import copyfile
import tempfile
import unittest

from QuantStudio import __QS_MainPath__
from QuantStudio.Factor.BaoStockDB import BaoStockDB


class TestBaoStockDBConfig(unittest.TestCase):
    def test_custom_info_file_accepts_pydantic_path(self):
        with tempfile.TemporaryDirectory() as directory:
            info_file = Path(directory) / "BaoStockDBInfo.xlsx"
            copyfile(Path(__QS_MainPath__) / "Resource/BaoStockDBInfo.xlsx", info_file)
            database = BaoStockDB(args={"DBInfoFile": str(info_file)})
            self.assertIsInstance(database.Args.DBInfoFile, Path)
            self.assertIn("A股K线数据", database.TableNames)
            self.assertIn("close", database.getTable("A股K线数据").FactorNames)


if __name__ == "__main__":
    unittest.main()
