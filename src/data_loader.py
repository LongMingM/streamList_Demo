"""
数据加载模块
负责文件系统操作和数据加载
"""

import os
import pandas as pd
import streamlit as st
from typing import List, Optional


class DataLoader:
    """数据加载器类"""
    
    def __init__(self):
        """初始化数据加载器"""
        pass
    
    def is_valid_directory(self, directory_path: str) -> bool:
        """
        检查目录是否有效
        
        Args:
            directory_path: 目录路径
            
        Returns:
            bool: 目录是否有效
        """
        return os.path.isdir(directory_path)
    
    def get_csv_files(self, directory_path: str) -> List[str]:
        """
        获取目录中的CSV文件列表
        
        Args:
            directory_path: 目录路径
            
        Returns:
            List[str]: CSV文件名列表
        """
        if not self.is_valid_directory(directory_path):
            return []
        
        try:
            file_list = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
            return file_list
        except Exception as e:
            st.error(f"读取目录时出错: {e}")
            return []
    
    def load_data(self, directory_path: str, file_name: str) -> Optional[pd.DataFrame]:
        """
        加载CSV数据文件
        
        Args:
            directory_path: 目录路径
            file_name: 文件名
            
        Returns:
            Optional[pd.DataFrame]: 加载的数据框，失败时返回None
        """
        try:
            file_path = os.path.join(directory_path, file_name)
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            st.error(f"加载文件时出错: {e}")
            return None
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        获取数据基本信息
        
        Args:
            df: 数据框
            
        Returns:
            dict: 数据信息字典
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        }
        return info 