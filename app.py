"""
数据可视化与机器学习预测分析工具
简化版 - 所有功能整合到一个文件
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, 
                            classification_report, f1_score, precision_score, 
                            recall_score, roc_auc_score, mean_absolute_error,
                            confusion_matrix, roc_curve, auc)
from typing import Optional, List, Dict, Any, Union
import warnings
warnings.filterwarnings('ignore')

# 导入配置
from config import APP_CONFIG, DEFAULT_PATHS

# 添加数据库支持
import sqlite3
import pymysql
import sqlalchemy
from sqlalchemy import create_engine, text

# 尝试导入XGBoost（可选）
XGBRegressor = None
XGBClassifier = None
try:
    from xgboost import XGBRegressor, XGBClassifier
    print("成功导入XGBoost")
except ImportError:
    print("XGBoost未安装或导入失败，将不使用XGBoost模型")
    XGBRegressor = None
    XGBClassifier = None

# 不在顶层导入LightGBM，而是在需要时导入
# LightGBM会在get_regression_models和get_classification_models函数中尝试导入

# ========== 工具函数 ==========
def is_valid_directory(directory_path: str) -> bool:
    """检查目录是否有效"""
    return os.path.isdir(directory_path)

def get_csv_files(directory_path: str) -> List[str]:
    """获取目录中的CSV文件列表"""
    if not is_valid_directory(directory_path):
        return []
    
    try:
        file_list = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
        return file_list
    except Exception as e:
        st.error(f"读取目录时出错: {e}")
        return []

def load_data(directory_path: str, file_name: str) -> Optional[pd.DataFrame]:
    """加载CSV数据文件"""
    try:
        file_path = os.path.join(directory_path, file_name)
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"加载文件时出错: {e}")
        return None

# 数据库连接和数据加载功能
def get_db_connection(db_type: str, host: str, port: int, 
                   username: str, password: str, database: str) -> Optional[Any]:
    """创建数据库连接"""
    try:
        if db_type == "mysql":
            connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
            engine = create_engine(connection_string)
            return engine
        elif db_type == "sqlite":
            # SQLite连接，数据库参数为数据库文件路径
            if db_type == "sqlite" and not os.path.exists(database):
                # 对于SQLite，如果文件不存在但目录有效，创建一个空数据库
                directory = os.path.dirname(database)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                    
            connection_string = f"sqlite:///{database}"
            engine = create_engine(connection_string)
            return engine
        else:
            st.error(f"不支持的数据库类型: {db_type}")
            return None
    except Exception as e:
        st.error(f"连接数据库时出错: {e}")
        return None

def get_tables_from_db(connection) -> List[str]:
    """获取数据库中的表列表"""
    try:
        # 检查连接类型并获取表列表
        if connection is not None:
            with connection.connect() as conn:
                if 'sqlite' in connection.url.drivername:
                    query = text("SELECT name FROM sqlite_master WHERE type='table';")
                else:
                    query = text("SHOW TABLES;")
                result = conn.execute(query)
                tables = [row[0] for row in result]
            return tables
        return []
    except Exception as e:
        st.error(f"获取表列表时出错: {e}")
        return []

def load_data_from_db(connection, query: str) -> Optional[pd.DataFrame]:
    """从数据库加载数据"""
    try:
        if connection is not None:
            df = pd.read_sql(query, connection)
            return df
        return None
    except Exception as e:
        st.error(f"从数据库加载数据时出错: {e}")
        return None

def remove_outliers(df, cols, method='3sigma'):
    """移除异常值"""
    df_clean = df.copy()
    if method == '3sigma':
        for col in cols:
            if df_clean[col].dtype in [np.float64, np.int64]:
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                df_clean = df_clean[(df_clean[col] >= mean - 3*std) & (df_clean[col] <= mean + 3*std)]
    elif method == 'iqr':
        for col in cols:
            if df_clean[col].dtype in [np.float64, np.int64]:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                df_clean = df_clean[(df_clean[col] >= Q1 - 1.5*IQR) & (df_clean[col] <= Q3 + 1.5*IQR)]
    return df_clean

# ========== 数据可视化模块 ==========
def render_visualization(df):
    """渲染可视化界面"""
    st.write("数据预览：")
    # 表格美化
    st.dataframe(df, use_container_width=True)

    st.write("数据描述：")
    st.dataframe(df.describe(), use_container_width=True)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if len(numeric_cols) > 0:
        # 使用会话状态来管理多选下拉框的状态
        viz_cols_key = 'viz_selected_cols'
        form_key = 'viz_cols_form'
        
        # 初始化会话状态（如果需要）
        if viz_cols_key not in st.session_state:
            st.session_state[viz_cols_key] = numeric_cols[:1] if numeric_cols else []
        
        # 过滤掉不在当前numeric_cols中的列
        st.session_state[viz_cols_key] = [col for col in st.session_state[viz_cols_key] if col in numeric_cols]
        
        # 直接定义回调函数，用于更新列表值而不重新渲染
        def update_viz_selection(selected_values):
            st.session_state[viz_cols_key] = selected_values
            
        # 使用容器包装选择器，减少重新渲染的影响
        selection_container = st.container()
        with selection_container:
            selected_cols = st.multiselect(
                "选择要可视化的数值列（可多选）", 
                options=numeric_cols,
                default=st.session_state[viz_cols_key],
                key=viz_cols_key
            )
        
        # 管理图表类型选择的会话状态
        chart_type_key = 'viz_chart_type'
        chart_options = ["折线图", "箱线图", "直方图", "相关性热力图"]
        
        # 初始化会话状态
        if chart_type_key not in st.session_state:
            st.session_state[chart_type_key] = "折线图"
        
        # 使用容器包装选择器
        chart_container = st.container()
        with chart_container:
            # 安全地计算默认索引
            default_index = 0
            if st.session_state[chart_type_key] in chart_options:
                default_index = chart_options.index(st.session_state[chart_type_key])
            
            chart_type = st.selectbox(
                "选择图表类型", 
                options=chart_options, 
                index=default_index,
                key=chart_type_key
            )
        
        if selected_cols:
            create_chart(df, selected_cols, chart_type)
        else:
            st.info("请至少选择一个数值列进行可视化。")
    else:
        st.warning("数据中没有数值列！")

def create_chart(df, selected_cols, chart_type):
    """创建指定类型的图表"""
    if chart_type == "折线图":
        fig = px.line(df, y=selected_cols, title="折线图", markers=True)
        fig.update_layout(xaxis_title="索引", yaxis_title=", ".join(selected_cols))
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "箱线图":
        if len(selected_cols) == 1:
            fig = px.box(df, y=selected_cols[0], title="箱线图")
        else:
            df_melt = df[selected_cols].melt(var_name="特征", value_name="值")
            fig = px.box(df_melt, x="特征", y="值", title="箱线图")
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "直方图":
        if len(selected_cols) > 1:
            fig = go.Figure()
            for col in selected_cols:
                fig.add_trace(go.Histogram(
                    x=df[col],
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
            fig = px.histogram(
                df, 
                x=selected_cols[0], 
                title="直方图",
                marginal="box",
                nbins=30
            )
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "相关性热力图":
        corr_matrix = pd.DataFrame(df[selected_cols]).corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="相关性热力图")
        st.plotly_chart(fig, use_container_width=True)

# ========== 特征工程模块 ==========
def render_feature_engineering(df):
    """渲染特征工程界面"""
    st.header("特征工程")
    
    # 数据预处理
    st.subheader("1. 数据预处理")
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        st.write("缺失值统计：")
        st.write(missing_data[missing_data > 0])
        
        # 管理缺失值处理策略选择的会话状态
        missing_key = 'missing_strategy'
        missing_options = ["删除行", "均值填充", "中位数填充", "众数填充"]
        
        # 初始化会话状态
        if missing_key not in st.session_state:
            st.session_state[missing_key] = "删除行"
        
        # 使用容器包装选择器
        missing_container = st.container()
        with missing_container:
            # 安全地计算默认索引
            default_index = 0
            if st.session_state[missing_key] in missing_options:
                default_index = missing_options.index(st.session_state[missing_key])
            
            missing_strategy = st.selectbox(
                "选择缺失值处理策略", 
                options=missing_options,
                index=default_index,
                key=missing_key
            )
        
        if missing_strategy == "删除行":
            df_clean = df.dropna()
        elif missing_strategy == "均值填充":
            df_clean = df.fillna(df.mean())
        elif missing_strategy == "中位数填充":
            df_clean = df.fillna(df.median())
        elif missing_strategy == "众数填充":
            df_clean = df.fillna(df.mode().iloc[0])
    else:
        df_clean = df.copy()
        st.success("数据中没有缺失值！")
    
    # 异常值处理
    st.subheader("2. 异常值处理")
    numeric_cols = df_clean.select_dtypes(include='number').columns.tolist()
    if len(numeric_cols) > 0:
        # 使用会话状态来管理异常值处理列选择
        outlier_cols_key = 'outlier_cols'
        
        # 初始化会话状态
        if outlier_cols_key not in st.session_state:
            st.session_state[outlier_cols_key] = []
        
        # 过滤掉不在numeric_cols中的列
        st.session_state[outlier_cols_key] = [col for col in st.session_state[outlier_cols_key] if col in numeric_cols]
        
        # 使用容器包装选择器，减少重新渲染的影响
        outlier_container = st.container()
        with outlier_container:
            outlier_cols = st.multiselect(
                "选择要处理异常值的列", 
                options=numeric_cols,
                default=st.session_state[outlier_cols_key],
                key=outlier_cols_key
            )
        
        if outlier_cols:
            # 添加可视化选项卡，展示异常值分布
            outlier_tabs = st.tabs(["异常值可视化", "异常值处理"])
            
            with outlier_tabs[0]:
                # 管理可视化方式选择的会话状态
                viz_method_key = 'viz_outlier_method'
                viz_method_options = ["箱线图", "直方图", "密度分布图"]
                
                # 初始化会话状态
                if viz_method_key not in st.session_state:
                    st.session_state[viz_method_key] = "箱线图"
                
                # 使用容器包装选择器
                viz_method_container = st.container()
                with viz_method_container:
                    # 安全地计算默认索引
                    default_index = 0
                    if st.session_state[viz_method_key] in viz_method_options:
                        default_index = viz_method_options.index(st.session_state[viz_method_key])
                    
                    viz_method = st.radio(
                        "选择可视化方式",
                        options=viz_method_options,
                        index=default_index,
                        key=viz_method_key,
                        horizontal=True
                    )
                
                # 选择要可视化的列，使用会话状态来保持选择
                viz_outlier_col_key = 'viz_outlier_col'
                
                # 初始化会话状态
                if viz_outlier_col_key not in st.session_state or st.session_state[viz_outlier_col_key] not in outlier_cols:
                    st.session_state[viz_outlier_col_key] = outlier_cols[0] if outlier_cols else None
                
                # 使用容器包装选择器
                viz_col_container = st.container()
                with viz_col_container:
                    # 安全地计算默认索引
                    default_index = 0
                    if st.session_state[viz_outlier_col_key] in outlier_cols:
                        default_index = outlier_cols.index(st.session_state[viz_outlier_col_key])
                    elif outlier_cols:
                        # 如果当前选择不在列表中，重置为第一个选项
                        st.session_state[viz_outlier_col_key] = outlier_cols[0]
                    
                    viz_col = st.selectbox(
                        "选择要可视化的列", 
                        options=outlier_cols,
                        index=default_index,
                        key=viz_outlier_col_key
                    )
                
                if viz_col:
                    # 计算统计信息，用于标记异常值
                    col_data = df_clean[viz_col]
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    iqr_lower = q1 - 1.5 * iqr
                    iqr_upper = q3 + 1.5 * iqr
                    
                    mean = col_data.mean()
                    std = col_data.std()
                    sigma3_lower = mean - 3 * std
                    sigma3_upper = mean + 3 * std
                    
                    # 标记异常值
                    iqr_outliers = df_clean[(col_data < iqr_lower) | (col_data > iqr_upper)][viz_col]
                    sigma_outliers = df_clean[(col_data < sigma3_lower) | (col_data > sigma3_upper)][viz_col]
                    
                    # 显示统计信息
                    stats_col1, stats_col2 = st.columns(2)
                    with stats_col1:
                        st.write("**IQR方法:**")
                        st.write(f"- Q1 (25%): {q1:.2f}")
                        st.write(f"- Q3 (75%): {q3:.2f}")
                        st.write(f"- IQR: {iqr:.2f}")
                        st.write(f"- 下界: {iqr_lower:.2f}")
                        st.write(f"- 上界: {iqr_upper:.2f}")
                        st.write(f"- 异常值数量: {len(iqr_outliers)}")
                    
                    with stats_col2:
                        st.write("**3sigma方法:**")
                        st.write(f"- 均值: {mean:.2f}")
                        st.write(f"- 标准差: {std:.2f}")
                        st.write(f"- 下界: {sigma3_lower:.2f}")
                        st.write(f"- 上界: {sigma3_upper:.2f}")
                        st.write(f"- 异常值数量: {len(sigma_outliers)}")
                    
                    # 根据选择的可视化方式展示图表
                    if viz_method == "箱线图":
                        fig = px.box(df_clean, y=viz_col, title=f"{viz_col}的箱线图")
                        
                        # 增加异常值标记
                        outlier_points = df_clean[
                            (df_clean[viz_col] < iqr_lower) | 
                            (df_clean[viz_col] > iqr_upper)
                        ][viz_col]
                        
                        if not outlier_points.empty:
                            fig.add_trace(go.Scatter(
                                y=outlier_points,
                                x=[0] * len(outlier_points),
                                mode='markers',
                                marker=dict(color='red', size=8, symbol='circle-open'),
                                name='IQR异常值'
                            ))
                        
                        # 添加边界线标记
                        fig.add_hline(y=iqr_lower, line_dash="dash", line_color="orange", 
                                      annotation_text="IQR下界", annotation_position="left")
                        fig.add_hline(y=iqr_upper, line_dash="dash", line_color="orange", 
                                      annotation_text="IQR上界", annotation_position="left")
                        fig.add_hline(y=sigma3_lower, line_dash="dot", line_color="red", 
                                      annotation_text="3σ下界", annotation_position="right")
                        fig.add_hline(y=sigma3_upper, line_dash="dot", line_color="red", 
                                      annotation_text="3σ上界", annotation_position="right")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif viz_method == "直方图":
                        fig = px.histogram(df_clean, x=viz_col, title=f"{viz_col}的分布直方图", 
                                          marginal="box", nbins=30)
                        
                        # 添加边界线标记
                        fig.add_vline(x=iqr_lower, line_dash="dash", line_color="orange", 
                                     annotation_text="IQR下界", annotation_position="top")
                        fig.add_vline(x=iqr_upper, line_dash="dash", line_color="orange", 
                                     annotation_text="IQR上界", annotation_position="top")
                        fig.add_vline(x=sigma3_lower, line_dash="dot", line_color="red", 
                                     annotation_text="3σ下界", annotation_position="bottom")
                        fig.add_vline(x=sigma3_upper, line_dash="dot", line_color="red", 
                                     annotation_text="3σ上界", annotation_position="bottom")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:  # 密度分布图
                        # 创建密度分布图
                        fig = go.Figure()
                        # 使用KDE估计密度
                        kde = stats.gaussian_kde(df_clean[viz_col].dropna())
                        x_vals = np.linspace(df_clean[viz_col].min(), df_clean[viz_col].max(), 1000)
                        y_vals = kde(x_vals)
                        
                        fig.add_trace(go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode='lines',
                            fill='tozeroy',
                            name='密度分布'
                        ))
                        
                        # 标记异常值区域
                        fig.add_vline(x=iqr_lower, line_dash="dash", line_color="orange", 
                                     annotation_text="IQR下界", annotation_position="top")
                        fig.add_vline(x=iqr_upper, line_dash="dash", line_color="orange", 
                                     annotation_text="IQR上界", annotation_position="top")
                        fig.add_vline(x=sigma3_lower, line_dash="dot", line_color="red", 
                                     annotation_text="3σ下界", annotation_position="bottom")
                        fig.add_vline(x=sigma3_upper, line_dash="dot", line_color="red", 
                                     annotation_text="3σ上界", annotation_position="bottom")
                        
                        fig.update_layout(
                            title=f"{viz_col}的密度分布图",
                            xaxis_title=viz_col,
                            yaxis_title="密度",
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            with outlier_tabs[1]:
                # 管理异常值处理方法选择的会话状态
                method_key = 'outlier_method'
                method_options = ["3sigma", "iqr"]
                
                # 初始化会话状态
                if method_key not in st.session_state:
                    st.session_state[method_key] = "3sigma"
                
                # 使用容器包装选择器
                method_container = st.container()
                with method_container:
                    # 安全地计算默认索引
                    default_index = 0
                    if st.session_state[method_key] in method_options:
                        default_index = method_options.index(st.session_state[method_key])
                    
                    outlier_method = st.selectbox(
                        "选择异常值处理方法", 
                        options=method_options,
                        index=default_index,
                        key=method_key
                    )
                
                if st.button("处理异常值"):
                    df_clean = remove_outliers(df_clean, outlier_cols, outlier_method)
                    st.success(f"已处理异常值，处理前数据量: {len(df)}，处理后数据量: {len(df_clean)}")
                    
                    # 添加处理前后对比
                    st.write("处理前后数据对比:")
                    
                    for col in outlier_cols:
                        st.write(f"### {col} 的异常值处理对比")
                        
                        # 创建两列布局
                        comp_col1, comp_col2 = st.columns(2)
                        
                        # 获取原数据和处理后数据的统计信息
                        orig_stats = df[col].describe()
                        clean_stats = df_clean[col].describe()
                        
                        stats_df = pd.DataFrame({
                            '原始数据': orig_stats,
                            '处理后数据': clean_stats
                        })
                        
                        # 显示统计信息表格
                        st.write(f"**{col}** 的统计信息:")
                        st.dataframe(stats_df.round(2), use_container_width=True)
                        
                        # 显示处理前后的箱线图对比
                        with comp_col1:
                            st.write("**处理前的分布:**")
                            fig_before = px.box(df, y=col, title="处理前")
                            st.plotly_chart(fig_before, use_container_width=True)
                            
                        with comp_col2:
                            st.write("**处理后的分布:**")
                            fig_after = px.box(df_clean, y=col, title="处理后")
                            st.plotly_chart(fig_after, use_container_width=True)
                        
                        # 添加直方图对比
                        fig_hist = go.Figure()
                        
                        fig_hist.add_trace(go.Histogram(
                            x=df[col],
                            name='处理前',
                            opacity=0.7,
                            nbinsx=30
                        ))
                        
                        fig_hist.add_trace(go.Histogram(
                            x=df_clean[col],
                            name='处理后',
                            opacity=0.7,
                            nbinsx=30
                        ))
                        
                        fig_hist.update_layout(
                            title=f"{col} 处理前后分布对比",
                            barmode='overlay',
                            xaxis_title=col,
                            yaxis_title="频次"
                        )
                        
                        st.plotly_chart(fig_hist, use_container_width=True)
    
    # 特征选择
    st.subheader("3. 特征和目标变量选择")
    numeric_cols = df_clean.select_dtypes(include='number').columns.tolist()
    if len(numeric_cols) == 0:
        st.warning("数据中没有数值列，无法进行机器学习预测！")
        target_col = None
        selected_features = []
    else:
        # 使用target_col_widget作为key，而不是target_col
        # 从session_state恢复之前选择的目标变量
        default_target_idx = 0
        if 'target_col_value' in st.session_state and st.session_state['target_col_value'] in numeric_cols:
            default_target_idx = numeric_cols.index(st.session_state['target_col_value'])
            
        target_col = st.selectbox(
            "选择目标变量（要预测的列）", 
            numeric_cols, 
            index=default_target_idx,
            key="target_col_widget"
        )
        # 保存选择值到session_state，而不是用widget的key
        st.session_state['target_col_value'] = target_col
        
        if target_col:
            corr_target = df_clean[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
            st.write("特征与目标变量的相关性排序：")
            st.dataframe(corr_target)
            recommended_features = [col for col in corr_target.index if col != target_col and corr_target[col] > 0.1]
            available_features = [col for col in numeric_cols if col != target_col]
            
            # 修复这里的错误，确保默认值是列表
            default_features = []
            if recommended_features:
                default_features = recommended_features[:min(5, len(recommended_features))]
            elif available_features:
                default_features = available_features[:min(5, len(available_features))]
                
            # 管理特征选择的会话状态
            feature_cols_key = 'feature_cols_value'
            
            # 初始化会话状态
            if feature_cols_key not in st.session_state:
                st.session_state[feature_cols_key] = []
            
            # 确保所有特征都在available_features列表中
            st.session_state[feature_cols_key] = [f for f in st.session_state[feature_cols_key] if f in available_features]
            
            # 如果没有预设选择，使用推荐特征
            if not st.session_state[feature_cols_key] and available_features:
                if recommended_features:
                    st.session_state[feature_cols_key] = recommended_features[:min(5, len(recommended_features))]
                else:
                    st.session_state[feature_cols_key] = available_features[:min(5, len(available_features))]
            
            # 使用容器包装特征选择器
            features_container = st.container()
            with features_container:
                selected_features = st.multiselect(
                    "选择特征变量（优先选择高相关性特征）", 
                    options=available_features, 
                    default=st.session_state[feature_cols_key],
                    key=feature_cols_key
                )
        else:
            selected_features = []
    
    # 保存处理后的数据和选择到session_state
    st.session_state['df_clean'] = df_clean
    st.session_state['target_col'] = target_col
    st.session_state['selected_features'] = selected_features
    
    return df_clean, target_col, selected_features

# ========== 模型预测模块 ==========
def render_model_prediction():
    """渲染模型预测界面"""
    st.header("机器学习模型训练")
    
    if 'df_clean' not in st.session_state or 'target_col' not in st.session_state or 'selected_features' not in st.session_state:
        st.info("请先在'特征工程'标签页完成数据处理")
        return
    
    df_clean = st.session_state['df_clean']
    target_col = st.session_state['target_col']
    selected_features = st.session_state['selected_features']
    
    if not target_col or not selected_features:
        st.info("请先在'特征工程'标签页选择目标变量和特征变量")
        return
    
    # 准备特征和目标变量
    X = df_clean[selected_features].copy()
    y = df_clean[target_col].copy()
    
    # 特征缩放
    if 'scaler' not in st.session_state:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        st.session_state['scaler'] = scaler
        st.session_state['X_scaled'] = X_scaled
    else:
        scaler = st.session_state['scaler']
        X_scaled = st.session_state.get('X_scaled', scaler.transform(X))
    
    # 确定问题类型
    if 'problem_type' not in st.session_state:
        if y.dtype in ['int64', 'float64'] and len(y.unique()) > 10:
            problem_type = "回归"
            models = get_regression_models()
        else:
            problem_type = "分类"
            # 确保分类标签从0开始（对XGBoost等模型很重要）
            if y.min() != 0 and XGBClassifier is not None:
                st.info(f"注意：分类标签已从 {y.min()} 调整为从0开始，以兼容XGBoost等模型")
                y = y - y.min()
            models = get_classification_models()
        
        st.session_state['problem_type'] = problem_type
        st.session_state['models'] = models
        st.session_state['y'] = y
    else:
        problem_type = st.session_state['problem_type']
        models = st.session_state['models']
        y = st.session_state.get('y', y)
    
    st.write(f"检测到的问题类型: {problem_type}")
    
    # 模型选择和训练
    model_names = list(models.keys())
    
    # 管理模型选择的会话状态
    models_key = 'selected_models'
    
    # 初始化会话状态
    if models_key not in st.session_state:
        st.session_state[models_key] = model_names[:2] if len(model_names) >= 2 else model_names
    
    # 过滤掉不在model_names中的模型
    st.session_state[models_key] = [model for model in st.session_state[models_key] if model in model_names]
    
    # 使用容器包装模型选择器
    models_container = st.container()
    with models_container:
        selected_models = st.multiselect(
            "选择要训练的模型（可多选）", 
            options=model_names, 
            default=st.session_state[models_key],
            key=models_key
        )
    
    # 从session_state恢复之前选择的测试集比例
    test_size = st.slider("测试集比例", 0.1, 0.5, st.session_state.get('test_size', 0.2), 0.05)
    st.session_state['test_size'] = test_size
    
    # 添加参数优化选项，从session_state恢复之前的选择
    enable_param_search = st.checkbox("启用参数优化", value=st.session_state.get('enable_param_search', False), 
                                     help="使用网格搜索或随机搜索寻找最佳参数")
    st.session_state['enable_param_search'] = enable_param_search
    
    if enable_param_search:
        search_method = st.radio("参数搜索方法", ["网格搜索(GridSearchCV)", "随机搜索(RandomizedSearchCV)"], 
                               index=0 if st.session_state.get('search_method', "") == "网格搜索(GridSearchCV)" else 1)
        st.session_state['search_method'] = search_method
        
        n_iter = 10
        if search_method == "随机搜索(RandomizedSearchCV)":
            n_iter = st.slider("随机搜索迭代次数", 5, 50, st.session_state.get('n_iter', 10))
            st.session_state['n_iter'] = n_iter
            
        cv_folds = st.slider("交叉验证折数", 2, 10, st.session_state.get('cv_folds', 3))
        st.session_state['cv_folds'] = cv_folds
    
    # 添加交叉验证选项（即使不做参数搜索也可用）
    enable_cross_validation = st.checkbox("启用交叉验证评估", value=st.session_state.get('enable_cross_validation', False),
                                        help="使用K折交叉验证评估模型表现")
    st.session_state['enable_cross_validation'] = enable_cross_validation
    
    if enable_cross_validation and not enable_param_search:
        cv_folds = st.slider("交叉验证折数", 2, 10, st.session_state.get('cv_folds', 5))
        st.session_state['cv_folds'] = cv_folds
    
    # 显示之前的训练结果（如果有）
    if 'results' in st.session_state and st.session_state['results']:
        st.write("## 之前的训练结果：")
        
        # 创建不包含模型对象的结果副本用于显示
        display_results = []
        for result in st.session_state['results']:
            display_result = {k: v for k, v in result.items() if k != 'model' and k != 'model_name'}
            display_results.append(display_result)
            
        st.dataframe(pd.DataFrame(display_results))
        
        if 'best_model_name' in st.session_state and st.session_state['best_model_name']:
            st.success(f"推荐最佳模型：{st.session_state['best_model_name']}")
            if 'best_params' in st.session_state and st.session_state['best_params'] != "默认参数":
                st.write("最佳参数：")
                st.json(st.session_state['best_params'])
    
    if st.button("开始训练所有模型"):
        with st.spinner("正在训练所有模型..."):
            # 如果之前没有拆分数据，或者测试集比例改变了，重新拆分数据
            if ('X_train' not in st.session_state or 'X_test' not in st.session_state or 
                'y_train' not in st.session_state or 'y_test' not in st.session_state or
                abs(st.session_state.get('last_test_size', 0) - test_size) > 0.001):
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size, random_state=42, stratify=y if problem_type == "分类" and len(np.unique(y)) < 10 else None
                )
                
                # 保存训练和测试数据到session_state
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['last_test_size'] = test_size
            else:
                X_train = st.session_state['X_train']
                X_test = st.session_state['X_test']
                y_train = st.session_state['y_train']
                y_test = st.session_state['y_test']
            
            results = []
            best_score = None
            best_model = None
            best_model_name = None
            best_params = None
            
            # 存储所有训练的模型
            trained_models = {}
            
            # 交叉验证结果
            cv_results_all = []
            
            # 进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, name in enumerate(selected_models):
                status_text.text(f"训练模型 {i+1}/{len(selected_models)}: {name}")
                progress_bar.progress((i) / len(selected_models))
                
                base_model = models[name]
                try:
                    # 设置模型训练开始时间
                    import time
                    start_time = time.time()
                    
                    if enable_param_search:
                        # 根据模型类型获取参数网格
                        param_grid = get_param_grid(name, problem_type)
                        
                        if param_grid:  # 如果有可用的参数网格
                            if search_method == "网格搜索(GridSearchCV)":
                                search = GridSearchCV(
                                    estimator=base_model,
                                    param_grid=param_grid,
                                    cv=cv_folds,
                                    scoring='r2' if problem_type == "回归" else 'accuracy',
                                    n_jobs=-1 if hasattr(base_model, 'n_jobs') else None,
                                    return_train_score=True
                                )
                            else:  # 随机搜索
                                search = RandomizedSearchCV(
                                    estimator=base_model,
                                    param_distributions=param_grid,
                                    n_iter=n_iter,
                                    cv=cv_folds,
                                    scoring='r2' if problem_type == "回归" else 'accuracy',
                                    n_jobs=-1 if hasattr(base_model, 'n_jobs') else None,
                                    random_state=42,
                                    return_train_score=True
                                )
                            
                            # 执行参数搜索
                            with st.spinner(f"正在为 {name} 优化参数..."):
                                search.fit(X_train, y_train)
                            
                            # 获取最佳模型和参数
                            model = search.best_estimator_
                            current_params = search.best_params_
                            cv_score = search.best_score_
                            
                            # 存储交叉验证详细结果
                            cv_df = pd.DataFrame(search.cv_results_)
                            cv_df = cv_df[['mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']]
                            cv_df['model'] = name
                            cv_results_all.append(cv_df)
                            
                            st.success(f"{name} 最佳参数: {current_params}")
                        else:
                            st.info(f"{name} 没有可用的参数网格，使用默认参数")
                            model = base_model
                            model.fit(X_train, y_train)
                            current_params = "默认参数"
                            
                            # 如果启用了交叉验证，单独进行交叉验证
                            if enable_cross_validation:
                                from sklearn.model_selection import cross_validate
                                cv_results = cross_validate(
                                    model, X_train, y_train, 
                                    cv=cv_folds,
                                    scoring='r2' if problem_type == "回归" else 'accuracy',
                                    return_train_score=True
                                )
                                cv_score = np.mean(cv_results['test_score'])
                                cv_df = pd.DataFrame({
                                    'mean_test_score': [cv_score],
                                    'std_test_score': [np.std(cv_results['test_score'])],
                                    'mean_train_score': [np.mean(cv_results['train_score'])],
                                    'std_train_score': [np.std(cv_results['train_score'])]
                                })
                                cv_df['model'] = name
                                cv_results_all.append(cv_df)
                            else:
                                cv_score = None
                    else:
                        model = base_model
                        model.fit(X_train, y_train)
                        current_params = "默认参数"
                        
                        # 如果启用了交叉验证，单独进行交叉验证
                        if enable_cross_validation:
                            from sklearn.model_selection import cross_validate
                            cv_results = cross_validate(
                                model, X_train, y_train, 
                                cv=cv_folds,
                                scoring='r2' if problem_type == "回归" else 'accuracy',
                                return_train_score=True
                            )
                            cv_score = np.mean(cv_results['test_score'])
                            cv_df = pd.DataFrame({
                                'mean_test_score': [cv_score],
                                'std_test_score': [np.std(cv_results['test_score'])],
                                'mean_train_score': [np.mean(cv_results['train_score'])],
                                'std_train_score': [np.std(cv_results['train_score'])]
                            })
                            cv_df['model'] = name
                            cv_results_all.append(cv_df)
                        else:
                            cv_score = None
                    
                    # 计算训练时间
                    training_time = time.time() - start_time
                    
                    # 测试集评估
                    y_pred = model.predict(X_test)
                    
                    # 计算训练集表现
                    y_train_pred = model.predict(X_train)
                    
                    # 保存训练好的模型
                    trained_models[name] = model
                    
                    if problem_type == "回归":
                        # 训练集评估
                        r2_train = r2_score(y_train, y_train_pred)
                        mse_train = mean_squared_error(y_train, y_train_pred)
                        
                        # 测试集评估
                        r2 = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)
                        
                        # 修复MAPE计算，确保使用数值而非列表
                        y_test_values = np.array(y_test)
                        mape = np.mean(np.abs((y_test_values - y_pred) / (np.maximum(0.0001, np.abs(y_test_values))))) * 100
                        
                        result = {
                            "模型": name,
                            "R²分数": f"{r2:.4f}",
                            "均方误差(MSE)": f"{mse:.4f}",
                            "均方根误差(RMSE)": f"{rmse:.4f}",
                            "平均绝对误差(MAE)": f"{mae:.4f}",
                            "平均绝对百分比误差(MAPE%)": f"{mape:.2f}",
                            "训练R²": f"{r2_train:.4f}",
                            "训练MSE": f"{mse_train:.4f}",
                            "CV得分": f"{cv_score:.4f}" if cv_score is not None else "N/A",
                            "训练时间(秒)": f"{training_time:.2f}",
                            "参数": str(current_params),
                            "model_name": name,
                            "model": model
                        }
                        results.append(result)
                        if best_score is None or r2 > best_score:
                            best_score = r2
                            best_model = model
                            best_model_name = name
                            best_params = current_params
                    else:
                        # 训练集评估
                        acc_train = accuracy_score(y_train, y_train_pred)
                        f1_train = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
                        
                        # 测试集评估
                        acc = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division='warn')
                        precision = precision_score(y_test, y_pred, average='weighted', zero_division='warn')
                        recall = recall_score(y_test, y_pred, average='weighted', zero_division='warn')
                        
                        # AUC只对二分类且有predict_proba支持的模型有效
                        auc_score = None
                        if hasattr(model, 'predict_proba') and len(np.unique(y_test)) == 2:
                            try:
                                auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                            except Exception:
                                auc_score = None
                        
                        result = {
                            "模型": name,
                            "准确率": f"{acc:.4f}",
                            "F1分数": f"{f1:.4f}",
                            "精确率": f"{precision:.4f}",
                            "召回率": f"{recall:.4f}",
                            "AUC": f"{auc_score:.4f}" if auc_score is not None else "N/A",
                            "训练准确率": f"{acc_train:.4f}",
                            "训练F1": f"{f1_train:.4f}",
                            "CV得分": f"{cv_score:.4f}" if cv_score is not None else "N/A",
                            "训练时间(秒)": f"{training_time:.2f}",
                            "参数": str(current_params),
                            "model_name": name,
                            "model": model
                        }
                        results.append(result)
                        if best_score is None or acc > best_score:
                            best_score = acc
                            best_model = model
                            best_model_name = name
                            best_params = current_params
                except Exception as e:
                    results.append({"模型": name, "错误": str(e)})
                
                # 更新进度条
                progress_bar.progress((i + 1) / len(selected_models))
            
            # 完成进度条
            progress_bar.progress(1.0)
            status_text.text("训练完成!")
            
            # 合并交叉验证结果
            if cv_results_all:
                cv_results_df = pd.concat(cv_results_all, ignore_index=True)
                st.session_state['cv_results'] = cv_results_df
            
            # 保存结果到session_state
            st.session_state['results'] = results
            st.session_state['best_score'] = best_score
            st.session_state['best_model'] = best_model
            st.session_state['best_model_name'] = best_model_name
            st.session_state['best_params'] = best_params
            st.session_state['trained_models'] = trained_models
            
            # 显示结果表格（不包含模型对象）
            display_results = []
            for result in results:
                display_result = {k: v for k, v in result.items() if k != 'model' and k != 'model_name'}
                display_results.append(display_result)
            
            st.write("## 各模型表现对比：")
            st.dataframe(pd.DataFrame(display_results))
            
            if best_model is not None:
                st.success(f"推荐最佳模型：{best_model_name}")
                if best_params != "默认参数":
                    st.write("最佳参数：")
                    st.json(best_params)
                
                # 添加学习曲线分析
                if st.checkbox("显示学习曲线分析", value=False):
                    with st.spinner("正在生成学习曲线..."):
                        render_learning_curves(best_model, X_scaled, y, best_model_name)
                
                # 添加特征重要性可视化
                if hasattr(best_model, 'feature_importances_'):
                    st.subheader("特征重要性")
                    importances = best_model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    feature_names = np.array(selected_features)[indices]
                    importances = importances[indices]
                    
                    fig = px.bar(
                        x=importances, 
                        y=feature_names,
                        orientation='h',
                        labels={'x': '重要性', 'y': '特征'},
                        title='特征重要性排序'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # 保存最佳模型到session_state
                st.session_state['model'] = best_model
                st.session_state['model_name'] = best_model_name
                st.session_state['y_pred'] = best_model.predict(X_test)

def get_regression_models():
    """获取回归模型字典"""
    models = {
        "线性回归": LinearRegression(),
        "岭回归": Ridge(),
        "Lasso回归": Lasso(),
        "决策树回归": DecisionTreeRegressor(),
        "随机森林回归": RandomForestRegressor(n_estimators=100, random_state=42),
        "支持向量回归": SVR(kernel='rbf')
    }
    if XGBRegressor is not None:
        models["XGBoost回归"] = XGBRegressor()
    
    # 在函数内部尝试导入LightGBM
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM回归"] = LGBMRegressor()
    except (ImportError, Exception) as e:
        st.write(f"LightGBM回归模型不可用: {e}")
    
    return models

def get_classification_models():
    """获取分类模型字典"""
    models = {
        "逻辑回归": LogisticRegression(random_state=42),
        "决策树分类": DecisionTreeClassifier(),
        "随机森林分类": RandomForestClassifier(n_estimators=100, random_state=42),
        "支持向量分类": SVC(kernel='rbf', probability=True, random_state=42),
        "K近邻分类": KNeighborsClassifier(),
        "朴素贝叶斯": GaussianNB()
    }
    if XGBClassifier is not None:
        models["XGBoost分类"] = XGBClassifier()
    
    # 在函数内部尝试导入LightGBM
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM分类"] = LGBMClassifier()
    except (ImportError, Exception) as e:
        st.write(f"LightGBM分类模型不可用: {e}")
    
    return models

def get_param_grid(model_name, problem_type):
    """根据模型名称和问题类型获取参数网格"""
    if "线性回归" in model_name:
        return {
            'fit_intercept': [True, False],
            'normalize': [True, False]
        }
    elif "岭回归" in model_name or "Lasso" in model_name:
        return {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'fit_intercept': [True, False]
        }
    elif "决策树" in model_name:
        return {
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif "随机森林" in model_name:
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif "支持向量" in model_name:
        return {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 1]
        }
    elif "K近邻" in model_name:
        return {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
    elif "逻辑回归" in model_name:
        return {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs'],
            'max_iter': [100, 200, 500]
        }
    elif "朴素贝叶斯" in model_name:
        return {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
    elif "XGBoost" in model_name:
        if problem_type == "回归":
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        else:
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
    elif "LightGBM" in model_name:
        if problem_type == "回归":
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'num_leaves': [31, 50, 100],
                'subsample': [0.8, 0.9, 1.0]
            }
        else:
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'num_leaves': [31, 50, 100],
                'subsample': [0.8, 0.9, 1.0]
            }
    return None  # 如果没有预定义的参数网格，返回None

# ========== 模型评估模块 ==========
def render_model_evaluation():
    """渲染模型评估界面"""
    st.header("模型评估与预测")
    
    if 'model' not in st.session_state:
        st.info("请先在'模型预测'标签页中训练模型。")
        return
    
    # 添加模型选择功能
    st.subheader("选择评估模型")
    
    # 检查是否有多个训练好的模型可供选择
    trained_models = {}
    if 'trained_models' in st.session_state:
        trained_models = st.session_state['trained_models']
    elif 'results' in st.session_state and st.session_state['results']:
        for result in st.session_state['results']:
            if 'model_name' in result and 'model' in result:
                model_name = result['model_name']
                trained_models[model_name] = result['model']
    
    # 如果没有找到存储的多个模型，使用当前最佳模型
    if not trained_models and 'model' in st.session_state:
        model_name = st.session_state.get('model_name', '当前最佳模型')
        trained_models[model_name] = st.session_state['model']
    
    # 如果有多个模型，显示选择框
    if len(trained_models) > 1:
        selected_model_name = st.selectbox(
            "选择要评估的模型", 
            list(trained_models.keys()),
            index=list(trained_models.keys()).index(st.session_state.get('model_name', list(trained_models.keys())[0])) if st.session_state.get('model_name') in trained_models else 0
        )
        
        # 更新当前选择的模型
        if selected_model_name != st.session_state.get('model_name'):
            st.session_state['model'] = trained_models[selected_model_name]
            st.session_state['model_name'] = selected_model_name
            
            # 更新预测结果
            if 'X_test' in st.session_state:
                st.session_state['y_pred'] = st.session_state['model'].predict(st.session_state['X_test'])
    else:
        selected_model_name = st.session_state.get('model_name', '当前模型')
        st.write(f"当前使用模型: **{selected_model_name}**")
    
    # 显示模型参数
    if selected_model_name:
        st.subheader("模型参数")
        model = st.session_state['model']
        
        # 尝试获取模型参数
        try:
            # 首先尝试从results中获取参数
            model_params = None
            if 'results' in st.session_state:
                for result in st.session_state['results']:
                    if result.get('模型') == selected_model_name and '参数' in result:
                        model_params = result['参数']
                        break
            
            # 如果没有找到，尝试从模型对象获取参数
            if model_params is None:
                if hasattr(model, 'get_params'):
                    model_params = model.get_params()
                else:
                    model_params = "无法获取模型参数"
            
            # 显示参数
            if isinstance(model_params, dict):
                st.json(model_params)
            else:
                st.write(model_params)
        except Exception as e:
            st.error(f"获取模型参数时出错: {e}")
    
    st.subheader("1. 模型性能评估")
    
    # 显示全面的模型评估指标
    render_model_metrics()
    
    # 显示测试集预测结果
    if st.session_state['problem_type'] == "回归":
        render_regression_evaluation()
    else:
        render_classification_evaluation()
    
    # 新数据预测界面
    st.subheader("2. 新数据预测")
    render_prediction_interface()

def render_model_metrics():
    """显示详细的模型评估指标"""
    if ('y_test' not in st.session_state or 'y_pred' not in st.session_state):
        st.warning("无法显示评估指标：缺少测试数据或预测结果")
        return

    y_test = st.session_state['y_test']
    y_pred = st.session_state['y_pred']
    
    # 创建两栏布局来显示指标
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        st.markdown("### 模型准确性指标")
        metrics_data = {}
        
        if st.session_state['problem_type'] == "回归":
            # 回归模型指标
            metrics_data = {
                "R²分数 (决定系数)": r2_score(y_test, y_pred),
                "均方误差 (MSE)": mean_squared_error(y_test, y_pred),
                "均方根误差 (RMSE)": np.sqrt(mean_squared_error(y_test, y_pred)),
                "平均绝对误差 (MAE)": mean_absolute_error(y_test, y_pred),
                "平均绝对百分比误差 (MAPE%)": np.mean(np.abs((y_test - y_pred) / (np.maximum(0.0001, np.abs(y_test))))) * 100
            }
            
            # 添加额外的回归评估
            explained_variance = np.var(y_pred) / np.var(y_test) if np.var(y_test) > 0 else 0
            metrics_data["解释方差比"] = explained_variance
            
            # 残差统计
            residuals = y_test - y_pred
            metrics_data["残差标准差"] = np.std(residuals)
            metrics_data["残差均值"] = np.mean(residuals)
            
        else:
            # 分类模型指标
            metrics_data = {
                "准确率 (Accuracy)": accuracy_score(y_test, y_pred),
                "F1分数 (加权平均)": f1_score(y_test, y_pred, average='weighted', zero_division='warn'),
                "精确率 (加权平均)": precision_score(y_test, y_pred, average='weighted', zero_division='warn'),
                "召回率 (加权平均)": recall_score(y_test, y_pred, average='weighted', zero_division='warn')
            }
            
            # 对于二分类问题，添加AUC-ROC
            if hasattr(st.session_state['model'], 'predict_proba') and len(np.unique(y_test)) == 2:
                try:
                    y_prob = st.session_state['model'].predict_proba(st.session_state['X_test'])[:, 1]
                    metrics_data["AUC-ROC"] = roc_auc_score(y_test, y_prob)
                except:
                    pass
        
        # 显示指标表格 - 修复DataFrame创建
        metrics_items = [(k, v) for k, v in metrics_data.items()]
        metrics_df = pd.DataFrame(metrics_items, columns=["指标", "值"])
        st.dataframe(metrics_df.style.format({"值": "{:.4f}"}), use_container_width=True)
    
    with metrics_col2:
        st.markdown("### 训练/测试数据信息")
        if 'X_train' in st.session_state and 'X_test' in st.session_state:
            data_info = {
                "训练集样本数": len(st.session_state['X_train']),
                "测试集样本数": len(st.session_state['X_test']),
                "特征数量": st.session_state['X_train'].shape[1],
                "测试集比例": st.session_state.get('last_test_size', 'N/A')
            }
            
            # 如果是分类问题，添加类别信息
            if st.session_state['problem_type'] == "分类":
                classes, counts = np.unique(y_test, return_counts=True)
                for i, c in enumerate(classes):
                    data_info[f"类别 {c}"] = counts[i]
            
            # 显示信息表格 - 修复DataFrame创建
            info_items = [(k, v) for k, v in data_info.items()]
            info_df = pd.DataFrame(info_items, columns=["信息", "值"])
            st.dataframe(info_df, use_container_width=True)
            
            # 添加交叉验证分数（如果可用）
            if 'cv_results' in st.session_state:
                st.markdown("### 交叉验证结果")
                st.dataframe(st.session_state['cv_results'], use_container_width=True)

def render_regression_evaluation():
    """渲染回归问题评估"""
    # 实际值vs预测值图
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=st.session_state['y_test'], 
        mode='markers', 
        name='实际值',
        marker=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        y=st.session_state['y_pred'], 
        mode='markers', 
        name='预测值',
        marker=dict(color='red')
    ))
    fig.update_layout(title="实际值 vs 预测值", xaxis_title="样本", yaxis_title="值")
    st.plotly_chart(fig, use_container_width=True)
    
    # 残差图
    residuals = st.session_state['y_test'] - st.session_state['y_pred']
    fig_residual = px.scatter(
        x=st.session_state['y_pred'], 
        y=residuals, 
        title="残差图", 
        labels={'x': '预测值', 'y': '残差'}
    )
    fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_residual, use_container_width=True)

def render_classification_evaluation():
    """渲染分类问题评估"""
    # 混淆矩阵
    cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('混淆矩阵')
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')
    st.pyplot(fig)
    
    # 添加分类报告可视化
    report = classification_report(
        st.session_state['y_test'], 
        st.session_state['y_pred'], 
        output_dict=True
    )
    
    # 转换为DataFrame以便可视化
    report_df = pd.DataFrame(report).T
    report_df = report_df.drop('support', axis=1)  # 移除support列以便更好地可视化
    
    # 使用热力图显示分类报告
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(report_df, annot=True, cmap='YlGnBu', ax=ax, fmt='.2f')
    ax.set_title('分类报告')
    st.pyplot(fig)
    
    # 如果模型支持预测概率，添加ROC曲线
    if (hasattr(st.session_state['model'], 'predict_proba') and 
        len(np.unique(st.session_state['y_test'])) == 2):
        y_prob = st.session_state['model'].predict_proba(st.session_state['X_test'])[:, 1]
        fpr, tpr, _ = roc_curve(st.session_state['y_test'], y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig = px.line(
            x=fpr, y=tpr,
            labels={'x': '假正例率', 'y': '真正例率'},
            title=f'ROC曲线 (AUC = {roc_auc:.3f})'
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        st.plotly_chart(fig, use_container_width=True)

def render_prediction_interface():
    """渲染预测界面"""
    # 创建预测输入表单
    st.write("输入新数据进行预测：")
    
    input_data = {}
    df_clean = st.session_state.get('df_clean')
    selected_features = st.session_state.get('selected_features', [])
    
    if df_clean is not None and selected_features:
        for feature in selected_features:
            # 获取该特征的数据范围用于设置默认值
            feature_data = df_clean[feature]
            default_value = float(feature_data.mean())
            min_val = float(feature_data.min())
            max_val = float(feature_data.max())
            
            # 计算步长，确保是浮点数
            diff = max_val - min_val
            step_val = float(diff / 100.0) if diff > 0 else 0.01
            
            input_data[feature] = st.number_input(
                f"{feature}",
                min_value=min_val,
                max_value=max_val,
                value=default_value,
                step=step_val
            )
        
        if st.button("进行预测"):
            make_prediction(input_data)

def make_prediction(input_data: dict):
    """进行预测"""
    try:
        # 准备输入数据
        input_df = pd.DataFrame([input_data])
        input_scaled = st.session_state['scaler'].transform(input_df)
        
        # 预测
        prediction = st.session_state['model'].predict(input_scaled)
        
        st.success(f"预测结果: {prediction[0]:.4f}")
        
        # 如果是分类问题，显示概率
        if (st.session_state['problem_type'] == "分类" and 
            hasattr(st.session_state['model'], 'predict_proba')):
            probabilities = st.session_state['model'].predict_proba(input_scaled)
            st.write("预测概率:")
            for i, prob in enumerate(probabilities[0]):
                st.write(f"类别 {i}: {prob:.4f}")
    except Exception as e:
        st.error(f"预测时出错: {e}")

def render_learning_curves(model, X, y, model_name):
    """生成并展示学习曲线"""
    from sklearn.model_selection import learning_curve
    
    # 生成学习曲线数据
    train_sizes = np.linspace(0.1, 1.0, 5)
    try:
        # 简化学习曲线计算，避免类型错误
        train_sizes, train_scores, val_scores = learning_curve(
            estimator=model,
            X=X,
            y=y,
            train_sizes=train_sizes,
            cv=5,
            scoring='r2' if st.session_state.get('problem_type', '') == "回归" else 'accuracy',
            n_jobs=-1 if hasattr(model, 'n_jobs') else None,
            random_state=42
        )
    
        # 计算平均值和标准差
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # 绘制学习曲线
        fig = go.Figure()
        
        # 训练集得分
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_mean,
            mode='lines+markers',
            name='训练集得分',
            line=dict(color='blue'),
            marker=dict(size=8)
        ))
        
        # 添加训练集标准差区域
        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(0,0,255,0.1)',
            line=dict(color='rgba(0,0,255,0)'),
            name='训练集标准差'
        ))
        
        # 测试集得分
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=val_mean,
            mode='lines+markers',
            name='验证集得分',
            line=dict(color='red'),
            marker=dict(size=8)
        ))
        
        # 添加测试集标准差区域
        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,0,0,0)'),
            name='验证集标准差'
        ))
        
        # 更新布局
        fig.update_layout(
            title=f"{model_name}学习曲线",
            xaxis_title="训练样本量",
            yaxis_title="得分",
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.5)'),
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 添加分析解释
        gap = np.mean(train_mean - val_mean)
        if gap > 0.1:
            st.warning(f"学习曲线分析：训练集和验证集性能差距较大 ({gap:.4f})，模型可能存在过拟合问题。")
        elif val_mean[-1] < 0.7:
            st.warning(f"学习曲线分析：验证集性能较低 ({val_mean[-1]:.4f})，模型可能存在欠拟合问题，需要更复杂的模型或更好的特征。")
        else:
            st.success(f"学习曲线分析：模型训练和验证性能良好，验证集最终得分 {val_mean[-1]:.4f}。")
        
        # 添加学习斜率分析
        if len(val_mean) >= 3:
            slope = (val_mean[-1] - val_mean[-3]) / (train_sizes[-1] - train_sizes[-3])
            if slope > 0.01:
                st.info("模型在增加更多训练数据时仍有提升空间，可考虑收集更多数据。")
            else:
                st.info("模型学习曲线已趋于平稳，增加更多数据可能效果有限。")
    except Exception as e:
        st.error(f"生成学习曲线时出错: {e}")
        return

# ========== 应用主流程 ==========
def main():
    st.set_page_config(
        page_title=APP_CONFIG['page_title'],
        page_icon=APP_CONFIG['page_icon'],
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 添加自定义CSS样式，使界面更像Kubeflow
    st.markdown("""
    <style>
    .card {
        border-radius: 10px;
        background-color: #f9f9f9;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 5px solid #4285F4;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .nav-button {
        margin-bottom: 10px;
    }
    .header-container {
        background-color: #4285F4;
        padding: 10px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .highlight {
        background-color: rgba(66, 133, 244, 0.1);
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #4285F4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 初始化session_state以存储当前功能
    if 'current_function' not in st.session_state:
        st.session_state['current_function'] = "数据可视化"
    
    # 创建两列布局，左侧为功能导航，右侧为功能内容
    left_col, right_col = st.columns([1, 4])
    
    with left_col:
        st.markdown('<div class="header-container"><h2 style="text-align:center;">ML工作流平台</h2></div>', unsafe_allow_html=True)
        
        # 数据加载部分放在左侧最上方，添加卡片样式
        st.markdown('<div class="card"><h3>数据源</h3>', unsafe_allow_html=True)
        
        # 添加数据源选择
        data_source = st.radio(
            "选择数据来源",
            ["本地数据", "数据库数据"],
            horizontal=True
        )
        
        # 根据选择的数据源显示不同的加载界面
        if data_source == "本地数据":
            DATA_DIR = st.text_input("数据文件夹路径", value=DEFAULT_PATHS['data_directory'])
            if is_valid_directory(DATA_DIR):
                file_list = get_csv_files(DATA_DIR)
                if not file_list:
                    st.info(f"目录 '{DATA_DIR}' 中没有找到CSV文件。")
            else:
                st.warning(f"目录 '{DATA_DIR}' 不存在。")
                file_list = []
            
            selected_file = st.selectbox("选择CSV文件", file_list, key="file_selector")
            
            # 当选择了文件时加载数据
            if selected_file:
                # 添加文件选择跟踪
                if 'previous_file' not in st.session_state or st.session_state['previous_file'] != selected_file:
                    st.session_state['previous_file'] = selected_file
                    # 清除之前的数据和相关状态
                    if 'df' in st.session_state:
                        del st.session_state['df']
                    if 'viz_selected_cols' in st.session_state:
                        del st.session_state['viz_selected_cols']
                    if 'viz_chart_type' in st.session_state:
                        del st.session_state['viz_chart_type']
                    
                # 加载新文件数据
                df = load_data(DATA_DIR, selected_file)
                data_source_info = f"本地文件: {selected_file}"
            else:
                df = None
                data_source_info = None
                
        else:  # 数据库数据
            # 初始化数据库连接参数
            if 'db_connection' not in st.session_state:
                st.session_state['db_connection'] = None
            
            # 创建两列布局用于数据库设置
            db_col1, db_col2 = st.columns(2)
            
            with db_col1:
                db_type = st.selectbox("数据库类型", ["mysql", "sqlite"], key="db_type")
                if db_type == "sqlite":
                    # 使用绝对路径确保SQLite能够正确找到文件
                    default_db_path = os.path.join(os.getcwd(), "data", "example.db")
                    database = st.text_input("数据库文件路径", key="db_database", value=default_db_path)
                    host = ""
                    port = 0
                    username = ""
                    password = ""
                else:
                    database = st.text_input("数据库名称", key="db_database", value="test")
            
            with db_col2:
                if db_type != "sqlite":
                    host = st.text_input("主机", key="db_host", value="localhost")
                    port = st.number_input("端口", key="db_port", value=3306, min_value=0, max_value=65535)
                    username = st.text_input("用户名", key="db_username", value="root")
                    password = st.text_input("密码", key="db_password", type="password")
            
            # 连接数据库按钮
            if st.button("连接数据库", key="connect_db"):
                connection = get_db_connection(db_type, host, port, username, password, database)
                if connection:
                    st.session_state['db_connection'] = connection
                    st.session_state['tables'] = get_tables_from_db(connection)
                    st.success(f"成功连接到数据库: {database}")
                else:
                    st.error("连接数据库失败")
            
            # 如果已连接数据库，显示表选择
            if st.session_state.get('db_connection'):
                tables = st.session_state.get('tables', [])
                if tables:
                    selected_table = st.selectbox("选择数据表", tables, key="table_selector")
                    custom_query = st.text_area(
                        "自定义SQL查询 (留空使用默认查询)",
                        value=f"SELECT * FROM {selected_table} LIMIT 1000" if tables else "",
                        height=100
                    )
                    
                    if st.button("加载数据", key="load_db_data"):
                        query = custom_query if custom_query else f"SELECT * FROM {selected_table} LIMIT 1000"
                        
                        # 跟踪当前查询，在查询变更时清除状态
                        current_query_id = f"{database}:{selected_table}:{query}"
                        if 'previous_query_id' not in st.session_state or st.session_state['previous_query_id'] != current_query_id:
                            st.session_state['previous_query_id'] = current_query_id
                            # 清除之前的数据和相关状态
                            if 'df' in st.session_state:
                                del st.session_state['df']
                            if 'viz_selected_cols' in st.session_state:
                                del st.session_state['viz_selected_cols']
                            if 'viz_chart_type' in st.session_state:
                                del st.session_state['viz_chart_type']
                        
                        # 加载新数据
                        df = load_data_from_db(st.session_state['db_connection'], query)
                        if df is not None:
                            st.session_state['df'] = df
                            data_source_info = f"数据库: {database}, 表: {selected_table}"
                            st.success(f"成功加载数据，共 {len(df)} 行")
                        else:
                            df = None
                            data_source_info = None
                else:
                    st.warning("数据库中没有表或无法获取表列表")
                    df = None
                    data_source_info = None
            else:
                df = None
                data_source_info = None
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 功能导航菜单放在左侧
        st.markdown('<div class="card"><h3>工作流组件</h3>', unsafe_allow_html=True)
        
        # 使用按钮作为导航项，添加样式类
        st.markdown('<div class="nav-button">', unsafe_allow_html=True)
        if st.button("📊 数据可视化", key="nav_viz", use_container_width=True, 
                  type="primary" if st.session_state['current_function'] == "数据可视化" else "secondary"):
            st.session_state['current_function'] = "数据可视化"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="nav-button">', unsafe_allow_html=True)
        if st.button("🛠️ 特征工程", key="nav_feat", use_container_width=True,
                  type="primary" if st.session_state['current_function'] == "特征工程" else "secondary"):
            st.session_state['current_function'] = "特征工程"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="nav-button">', unsafe_allow_html=True)
        if st.button("🤖 模型训练", key="nav_pred", use_container_width=True,
                  type="primary" if st.session_state['current_function'] == "模型训练" else "secondary"):
            st.session_state['current_function'] = "模型训练"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="nav-button">', unsafe_allow_html=True)
        if st.button("📈 模型评估", key="nav_eval", use_container_width=True,
                  type="primary" if st.session_state['current_function'] == "模型评估" else "secondary"):
            st.session_state['current_function'] = "模型评估"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 右侧内容区域
    with right_col:
        current_function = st.session_state['current_function']
        # 使用更像Kubeflow的标题栏
        st.markdown(f'<div class="header-container"><h1>{current_function}</h1></div>', unsafe_allow_html=True)
        
        # 获取当前数据，可能来自本地文件或数据库
        if 'df' in st.session_state and st.session_state['df'] is not None:
            df = st.session_state['df']  # 使用已存在的数据
        elif data_source == "本地数据" and selected_file:
            df = load_data(DATA_DIR, selected_file)
            if df is not None:
                # 确保session_state中的数据是最新的
                st.session_state['df'] = df
                # 如果是首次加载或文件变更，重置相关状态
                if 'previous_file_loaded' not in st.session_state or st.session_state['previous_file_loaded'] != selected_file:
                    st.session_state['previous_file_loaded'] = selected_file
                    # 清除特定功能的状态
                    for key in list(st.session_state.keys()):
                        if isinstance(key, str) and (key.startswith('viz_') or key.startswith('outlier_') or 
                                                key.startswith('missing_') or key.startswith('feature_')):
                            del st.session_state[key]
        else:
            df = None
        
        # 检查是否有可用数据
        if df is None:
            st.warning("请选择或加载数据！")
            return
        
        # 添加数据信息显示和刷新按钮
        col1, col2 = st.columns([10, 1])
        
        with col1:
            if data_source_info:
                info_text = f"数据源: <b>{data_source_info}</b> | 记录数: <b>{len(df)}</b> | 特征数: <b>{len(df.columns)}</b>"
            else:
                info_text = f"记录数: <b>{len(df)}</b> | 特征数: <b>{len(df.columns)}</b>"
                
            st.markdown(f'<div class="highlight">{info_text}</div>', unsafe_allow_html=True)
        
        with col2:
            # 添加刷新按钮
            if st.button("🔄", help="刷新数据和视图"):
                # 清除所有相关状态，强制重新加载
                for key in list(st.session_state.keys()):
                    if isinstance(key, str) and (key.startswith('viz_') or key.startswith('outlier_') or 
                                            key.startswith('missing_') or key.startswith('feature_')):
                        del st.session_state[key]
                # 强制重新运行
                st.rerun()
            
        # 根据当前选择的功能显示相应内容
        if current_function == "数据可视化":
            render_visualization(df)
        elif current_function == "特征工程":
            render_feature_engineering(df)
        elif current_function == "模型预测" or current_function == "模型训练":
            render_model_prediction()
        elif current_function == "模型评估":
            render_model_evaluation()

if __name__ == "__main__":
    main()
