"""因子表在活动计算上下文中的多因子读取回归。"""
from datetime import datetime, timedelta
import tempfile
import unittest

import pandas as pd

from QuantStudio.Core.CalcEngine import Engine, StackEngine
from QuantStudio.Core.QSObject import Panel
from QuantStudio.Factor.Factor import FactorContext
from QuantStudio.Factor.HDF5DB import HDF5DB


class TestFactorTable(unittest.TestCase):
    """真实临时因子库覆盖直接读取和引擎内读取两条路径。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = HDF5DB(args={"MainDir": self.directory.name}).connect()
        self.addCleanup(self.database.disconnect)
        self.dates = [datetime(2026, 1, 1) + timedelta(days=index) for index in range(3)]
        self.ids = ["A", "B"]
        self.data = pd.DataFrame([[1., 2.], [3., 4.], [5., 6.]], index=self.dates, columns=self.ids)
        self.database.writeData(Panel({"open": self.data, "close": self.data * 2}), "prices")

    def test_read_multiple_factors_with_active_context(self):
        """每个目标因子获得类型化初始化参数和对应的前向请求。"""
        for engine_type in (Engine, StackEngine):
            with self.subTest(engine=engine_type.__name__):
                with FactorContext(DTRuler=self.dates, SectionIDs=self.ids), engine_type(args={"IOConcurrentNum": 1}):
                    result = self.database.getTable("prices").readData(
                        factor_names=["close", "open"], ids=["B", "A"], dts=self.dates[1:], section_ids=self.ids)
                pd.testing.assert_frame_equal(result.iloc[0], (self.data * 2).loc[self.dates[1:], ["B", "A"]], check_index_type=False)
                pd.testing.assert_frame_equal(result.iloc[1], self.data.loc[self.dates[1:], ["B", "A"]], check_index_type=False)


if __name__ == "__main__":
    unittest.main()
