"""配置路径解析的回归测试，所有配置文件均在临时目录中创建。"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from QuantStudio.Core import __QS_Object__


class ConfigObject(__QS_Object__):
    """只读取一个整数参数的配置测试对象。"""

    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        Value: int = 0


class TestConfigFile(unittest.TestCase):
    """验证绝对路径、默认目录及显式参数优先级。"""

    def test_absolute_path_with_mixed_separators(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"Value": 3}), encoding="utf-8")
            obj = ConfigObject(config_file=str(path).replace(os.sep, "/"), args={"Value": 7})
            self.assertEqual(Path(obj.ConfigFile), path)
            self.assertEqual(obj.Args.Value, 7)

    def test_missing_absolute_path_is_not_prefixed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.json"
            logger = Mock()
            obj = ConfigObject(config_file=str(path).replace(os.sep, "/"), logger=logger)
            self.assertIsNone(obj.ConfigFile)
            logger.warning.assert_called_once_with(f"找不到配置文件: {os.path.normpath(path)}")

    def test_relative_path_falls_back_to_default_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            name = "default_config_regression.json"
            path = Path(directory) / name
            path.write_text(json.dumps({"Value": 5}), encoding="utf-8")
            with patch("QuantStudio.Core.__QS_ConfigPath__", directory):
                obj = ConfigObject(config_file=name)
            self.assertEqual(Path(obj.ConfigFile), path)
            self.assertEqual(obj.Args.Value, 5)


if __name__ == "__main__":
    unittest.main()
