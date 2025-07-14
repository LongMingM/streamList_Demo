"""
配置文件
包含应用的配置参数
"""

import os

# 应用配置
APP_CONFIG = {
    'page_title': '数据可视化与机器学习预测分析',
    'page_icon': '📊',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# 默认路径配置
DEFAULT_PATHS = {
    'data_directory': r"D:\code_study\ML_CODE\kaggle\Regression\Red Wine Quality"
}

# 模型配置
MODEL_CONFIG = {
    'regression_models': {
        'linear_regression': {
            'name': '线性回归',
            'params': {}
        },
        'random_forest_regression': {
            'name': '随机森林回归',
            'params': {'n_estimators': 100, 'random_state': 42}
        },
        'svr': {
            'name': '支持向量回归',
            'params': {'kernel': 'rbf'}
        }
    },
    'classification_models': {
        'logistic_regression': {
            'name': '逻辑回归',
            'params': {'random_state': 42}
        },
        'random_forest_classification': {
            'name': '随机森林分类',
            'params': {'n_estimators': 100, 'random_state': 42}
        },
        'svc': {
            'name': '支持向量分类',
            'params': {'kernel': 'rbf', 'random_state': 42}
        }
    }
}

# 数据预处理配置
PREPROCESSING_CONFIG = {
    'missing_value_strategies': [
        '删除行',
        '均值填充',
        '中位数填充',
        '众数填充'
    ],
    'test_size_range': (0.1, 0.5),
    'default_test_size': 0.2
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'chart_types': [
        '折线图',
        '箱线图',
        '直方图',
        '相关性热力图'
    ],
    'color_schemes': {
        'highlight_max': 'lightgreen',
        'highlight_min': 'lightcoral',
        'gradient_cmap': 'Blues'
    }
} 