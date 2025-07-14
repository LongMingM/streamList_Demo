"""
数据可视化模块
负责数据可视化和图表展示
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import List


class DataVisualizer:
    """数据可视化器类"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化可视化器
        
        Args:
            df: 数据框
        """
        self.df = df
        self.numeric_cols = df.select_dtypes(include='number').columns.tolist()
    
    def render(self):
        """渲染可视化界面"""
        st.write("数据预览：")
        # 表格美化
        styled_df = self.df.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightcoral')
        st.dataframe(styled_df, use_container_width=True)

        st.write("数据描述：")
        st.dataframe(self.df.describe().style.background_gradient(cmap='Blues'), use_container_width=True)

        if len(self.numeric_cols) > 0:
            selected_cols = st.multiselect(
                "选择要可视化的数值列（可多选）", 
                self.numeric_cols, 
                default=self.numeric_cols[:1] if self.numeric_cols else []
            )
            chart_type = st.selectbox("选择图表类型", ["折线图", "箱线图", "直方图", "相关性热力图"])
            
            if selected_cols:
                self._create_chart(selected_cols, chart_type)
            else:
                st.info("请至少选择一个数值列进行可视化。")
        else:
            st.warning("数据中没有数值列！")
    
    def _create_chart(self, selected_cols: List[str], chart_type: str):
        """
        创建指定类型的图表
        
        Args:
            selected_cols: 选中的列
            chart_type: 图表类型
        """
        if chart_type == "折线图":
            self._create_line_chart(selected_cols)
        elif chart_type == "箱线图":
            self._create_box_chart(selected_cols)
        elif chart_type == "直方图":
            self._create_histogram(selected_cols)
        elif chart_type == "相关性热力图":
            self._create_correlation_heatmap(selected_cols)
    
    def _create_line_chart(self, selected_cols: List[str]):
        """创建折线图"""
        fig = px.line(self.df, y=selected_cols, title="折线图", markers=True)
        fig.update_layout(xaxis_title="索引", yaxis_title=", ".join(selected_cols))
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_box_chart(self, selected_cols: List[str]):
        """创建箱线图"""
        if len(selected_cols) == 1:
            fig = px.box(self.df, y=selected_cols[0], title="箱线图")
        else:
            df_melt = self.df[selected_cols].melt(var_name="特征", value_name="值")
            fig = px.box(df_melt, x="特征", y="值", title="箱线图")
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_histogram(self, selected_cols: List[str]):
        """创建直方图"""
        if len(selected_cols) > 1:
            # 支持多列直方图比较
            fig = go.Figure()
            for col in selected_cols:
                fig.add_trace(go.Histogram(
                    x=self.df[col],
                    name=col,
                    opacity=0.7,
                    nbinsx=30
                ))
            fig.update_layout(
                title="多特征直方图比较",
                barmode='overlay',
                xaxis_title="值",
                yaxis_title="频次"
            )
        else:
            # 单列直方图
            fig = px.histogram(
                self.df, 
                x=selected_cols[0], 
                title="直方图",
                marginal="box",  # 添加箱线图在边缘
                nbins=30
            )
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_correlation_heatmap(self, selected_cols: List[str]):
        """创建相关性热力图"""
        corr_matrix = pd.DataFrame(self.df[selected_cols]).corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="相关性热力图")
        st.plotly_chart(fig, use_container_width=True) 