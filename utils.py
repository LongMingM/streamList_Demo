"""
工具函数模块
包含通用的工具函数
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import streamlit as st


def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """验证数据框的有效性"""
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }
    
    if df.empty:
        validation_result['is_valid'] = False
        validation_result['errors'].append("数据框为空")
        return validation_result
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        validation_result['warnings'].append("没有数值列，无法进行机器学习")
    
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        validation_result['warnings'].append(f"发现 {missing_data.sum()} 个缺失值")
    
    validation_result['info'] = {
        'shape': df.shape,
        'numeric_columns': numeric_cols,
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'missing_values': missing_data.to_dict()
    }
    
    return validation_result


def detect_problem_type(y: pd.Series) -> str:
    """检测问题类型（回归或分类）"""
    if y.dtype in ['int64', 'float64'] and len(y.unique()) > 10:
        return "回归"
    else:
        return "分类"


def format_number(value: float, decimal_places: int = 4) -> str:
    """格式化数字显示"""
    return f"{value:.{decimal_places}f}" 