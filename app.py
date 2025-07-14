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
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
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

# 应用配置
APP_CONFIG = {
    'page_title': '数据可视化与机器学习预测分析',
    'page_icon': '📊',
    'layout': "wide",  # 使用字面量而非变量
    'initial_sidebar_state': "expanded"  # 使用字面量而非变量
}

# 默认路径配置
DEFAULT_PATHS = {
    'data_directory': r"D:\code_study\ML_CODE\kaggle\Regression\Red Wine Quality"
}

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
        # 从session_state恢复之前选择的列
        default_cols = st.session_state.get('viz_selected_cols', numeric_cols[:1] if numeric_cols else [])
        selected_cols = st.multiselect(
            "选择要可视化的数值列（可多选）", 
            numeric_cols, 
            default=default_cols
        )
        # 保存选择到session_state
        st.session_state['viz_selected_cols'] = selected_cols
        
        # 从session_state恢复之前选择的图表类型
        default_chart_type = st.session_state.get('viz_chart_type', "折线图")
        chart_type = st.selectbox("选择图表类型", ["折线图", "箱线图", "直方图", "相关性热力图"], 
                                 index=["折线图", "箱线图", "直方图", "相关性热力图"].index(default_chart_type))
        # 保存选择到session_state
        st.session_state['viz_chart_type'] = chart_type
        
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
        
        # 从session_state恢复之前选择的缺失值处理策略
        default_missing_strategy = st.session_state.get('missing_strategy', "删除行")
        missing_strategy = st.selectbox(
            "选择缺失值处理策略", 
            ["删除行", "均值填充", "中位数填充", "众数填充"],
            index=["删除行", "均值填充", "中位数填充", "众数填充"].index(default_missing_strategy)
        )
        # 保存选择到session_state
        st.session_state['missing_strategy'] = missing_strategy
        
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
        # 从session_state恢复之前选择的异常值处理列
        default_outlier_cols = st.session_state.get('outlier_cols', [])
        outlier_cols = st.multiselect(
            "选择要处理异常值的列", 
            numeric_cols,
            default=default_outlier_cols
        )
        # 保存选择到session_state
        st.session_state['outlier_cols'] = outlier_cols
        
        if outlier_cols:
            # 从session_state恢复之前选择的异常值处理方法
            default_outlier_method = st.session_state.get('outlier_method', "3sigma")
            outlier_method = st.selectbox(
                "选择异常值处理方法", 
                ["3sigma", "iqr"],
                index=["3sigma", "iqr"].index(default_outlier_method)
            )
            # 保存选择到session_state
            st.session_state['outlier_method'] = outlier_method
            
            if st.button("处理异常值"):
                df_clean = remove_outliers(df_clean, outlier_cols, outlier_method)
                st.success(f"已处理异常值，处理前数据量: {len(df)}，处理后数据量: {len(df_clean)}")
    
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
                
            # 从session_state恢复之前选择的特征变量
            if 'feature_cols_value' in st.session_state:
                saved_features = st.session_state['feature_cols_value']
                default_features = [f for f in saved_features if f in available_features]
                
            # 使用feature_cols_widget作为key，而不是feature_cols
            selected_features = st.multiselect(
                "选择特征变量（优先选择高相关性特征）", 
                available_features, 
                default=default_features,
                key="feature_cols_widget"
            )
            # 保存选择值到session_state，而不是用widget的key
            st.session_state['feature_cols_value'] = selected_features
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
    st.header("机器学习模型预测")
    
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
    
    # 从session_state恢复之前选择的模型
    default_models = st.session_state.get('selected_models', model_names[:2])
    selected_models = st.multiselect("选择要训练的模型（可多选）", model_names, default=default_models)
    st.session_state['selected_models'] = selected_models
    
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
                    X_scaled, y, test_size=test_size, random_state=42
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
            
            for name in selected_models:
                base_model = models[name]
                try:
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
                                    n_jobs=-1 if hasattr(base_model, 'n_jobs') else None
                                )
                            else:  # 随机搜索
                                search = RandomizedSearchCV(
                                    estimator=base_model,
                                    param_distributions=param_grid,
                                    n_iter=n_iter,
                                    cv=cv_folds,
                                    scoring='r2' if problem_type == "回归" else 'accuracy',
                                    n_jobs=-1 if hasattr(base_model, 'n_jobs') else None,
                                    random_state=42
                                )
                            
                            # 执行参数搜索
                            with st.spinner(f"正在为 {name} 优化参数..."):
                                search.fit(X_train, y_train)
                            
                            # 获取最佳模型和参数
                            model = search.best_estimator_
                            current_params = search.best_params_
                            st.success(f"{name} 最佳参数: {current_params}")
                        else:
                            st.info(f"{name} 没有可用的参数网格，使用默认参数")
                            model = base_model
                            model.fit(X_train, y_train)
                            current_params = "默认参数"
                    else:
                        model = base_model
                        model.fit(X_train, y_train)
                        current_params = "默认参数"
                    
                    y_pred = model.predict(X_test)
                    
                    # 保存训练好的模型
                    trained_models[name] = model
                    
                    if problem_type == "回归":
                        r2 = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)
                        # 修复MAPE计算，确保使用数值而非列表
                        y_test_values = np.array(y_test)
                        mape = np.mean(np.abs((y_test_values - y_pred) / (y_test_values + 1e-8))) * 100
                        result = {
                            "模型": name,
                            "R²分数": f"{r2:.4f}",
                            "均方误差(MSE)": f"{mse:.4f}",
                            "均方根误差(RMSE)": f"{rmse:.4f}",
                            "平均绝对误差(MAE)": f"{mae:.4f}",
                            "平均绝对百分比误差(MAPE%)": f"{mape:.2f}",
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
                            "AUC": f"{auc_score:.4f}" if auc_score is not None else "-",
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
    
    # 显示测试集预测结果
    if st.session_state['problem_type'] == "回归":
        render_regression_evaluation()
    else:
        render_classification_evaluation()
    
    # 新数据预测界面
    st.subheader("2. 新数据预测")
    render_prediction_interface()

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

# ========== 应用主流程 ==========
def main():
    st.set_page_config(
        page_title=APP_CONFIG['page_title'],
        page_icon=APP_CONFIG['page_icon'],
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("数据可视化与机器学习预测分析")
    
    # 初始化session_state以存储当前标签页
    if 'current_tab' not in st.session_state:
        st.session_state['current_tab'] = 0
    
    # 侧边栏：数据加载
    DATA_DIR = st.sidebar.text_input("数据文件夹路径", value=DEFAULT_PATHS['data_directory'])
    if is_valid_directory(DATA_DIR):
        file_list = get_csv_files(DATA_DIR)
    else:
        file_list = []
    
    selected_file = st.sidebar.selectbox("选择文件", file_list, key="file_selector")
    
    if selected_file:
        df = load_data(DATA_DIR, selected_file)
        if df is None:
            st.error("无法加载数据文件！")
            return
        
        # 初始化session_state
        if 'df' not in st.session_state:
            st.session_state['df'] = df
        
        # 创建标签页，使用radio代替tabs以便更好地保持状态
        tab_options = ["📊 数据可视化", "🛠️ 特征工程", "🤖 模型预测", "📈 模型评估"]
        current_tab = st.radio("选择功能", tab_options, index=int(st.session_state['current_tab']), horizontal=True)
        
        # 更新当前标签页索引
        st.session_state['current_tab'] = tab_options.index(current_tab)
        
        # 根据选择的标签页显示相应内容
        if current_tab == "📊 数据可视化":
            render_visualization(df)
        elif current_tab == "🛠️ 特征工程":
            render_feature_engineering(df)
        elif current_tab == "🤖 模型预测":
            render_model_prediction()
        elif current_tab == "📈 模型评估":
            render_model_evaluation()
    
    else:
        st.warning(f"文件夹 '{DATA_DIR}' 不存在或没有csv文件！")

if __name__ == "__main__":
    main()
