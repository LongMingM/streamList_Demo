"""
应用测试文件
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from utils import validate_dataframe, detect_problem_type, format_number


class TestDataLoader(unittest.TestCase):
    """数据加载器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.data_loader = DataLoader()
        
        # 创建测试数据
        self.test_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [1.1, 2.2, 3.3, 4.4, 5.5],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_is_valid_directory(self):
        """测试目录有效性检查"""
        # 测试有效目录
        self.assertTrue(self.data_loader.is_valid_directory('.'))
        
        # 测试无效目录
        self.assertFalse(self.data_loader.is_valid_directory('nonexistent_directory'))
    
    def test_get_data_info(self):
        """测试数据信息获取"""
        info = self.data_loader.get_data_info(self.test_df)
        
        self.assertEqual(info['shape'], (5, 3))
        self.assertEqual(len(info['columns']), 3)
        self.assertEqual(len(info['numeric_columns']), 3)


class TestUtils(unittest.TestCase):
    """工具函数测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.test_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [1.1, 2.2, 3.3, 4.4, 5.5],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_validate_dataframe(self):
        """测试数据框验证"""
        result = validate_dataframe(self.test_df)
        
        self.assertTrue(result['is_valid'])
        self.assertEqual(len(result['errors']), 0)
        self.assertEqual(result['info']['shape'], (5, 3))
    
    def test_detect_problem_type(self):
        """测试问题类型检测"""
        # 测试分类问题
        y_classification = pd.Series([0, 1, 0, 1, 0])
        self.assertEqual(detect_problem_type(y_classification), "分类")
        
        # 测试回归问题
        y_regression = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.1])
        self.assertEqual(detect_problem_type(y_regression), "回归")
    
    def test_format_number(self):
        """测试数字格式化"""
        self.assertEqual(format_number(3.14159, 2), "3.14")
        self.assertEqual(format_number(3.14159, 4), "3.1416")


if __name__ == '__main__':
    unittest.main() 