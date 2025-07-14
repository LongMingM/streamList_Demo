"""
机器学习预测模块
负责数据预处理、模型训练和预测
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# XGBoost & LightGBM
try:
    from xgboost import XGBRegressor, XGBClassifier
except ImportError:
    XGBRegressor = None
    XGBClassifier = None
try:
    from lightgbm import LGBMRegressor, LGBMClassifier
except ImportError:
    LGBMRegressor = None
    LGBMClassifier = None

class MLPredictor:
    """机器学习预测器类"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.df_clean = None
        self.problem_type = None
        self.models = {}
        self.selected_models = []
        self.best_model_name = None
        self.best_model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
    
    def render(self):
        st.header("机器学习模型预测")
        self._render_data_preprocessing()
        self._render_feature_selection()
        if hasattr(self, 'selected_features') and self.selected_features:
            self._render_model_training()
    
    def _render_data_preprocessing(self):
        st.subheader("1. 数据预处理")
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            st.write("缺失值统计：")
            st.write(missing_data[missing_data > 0])
            missing_strategy = st.selectbox(
                "选择缺失值处理策略", 
                ["删除行", "均值填充", "中位数填充", "众数填充"]
            )
            if missing_strategy == "删除行":
                self.df_clean = self.df.dropna()
            elif missing_strategy == "均值填充":
                self.df_clean = self.df.fillna(self.df.mean())
            elif missing_strategy == "中位数填充":
                self.df_clean = self.df.fillna(self.df.median())
            elif missing_strategy == "众数填充":
                self.df_clean = self.df.fillna(self.df.mode().iloc[0])
        else:
            self.df_clean = self.df.copy()
            st.success("数据中没有缺失值！")
    
    def _render_feature_selection(self):
        st.subheader("2. 特征和目标变量选择")
        if self.df_clean is None:
            st.error("请先完成数据预处理！")
            return
        # 分离数值和分类特征
        numeric_features = self.df_clean.select_dtypes(include=[np.number]).columns.tolist() if self.df_clean is not None else []
        categorical_features = self.df_clean.select_dtypes(include=['object']).columns.tolist() if self.df_clean is not None else []
        if len(numeric_features) == 0:
            st.error("数据中没有数值特征，无法进行机器学习预测！")
            return
        
        # 只在这里渲染一次
        target_col = st.selectbox("请选择目标变量", numeric_features, key="ml_target_col")
        available_features = [col for col in numeric_features if col != target_col]
        if len(available_features) > 0:
            self.selected_features = st.multiselect(
                "请选择特征变量", 
                available_features, 
                default=available_features[:min(5, len(available_features))],
                key="ml_feature_cols"
            )
            if self.selected_features and target_col:
                self._prepare_data(target_col, categorical_features)
            else:
                st.warning("请选择特征变量！")
    
    def _prepare_data(self, target_col: str, categorical_features: list):
        # 添加检查确保df_clean和selected_features不为None
        if self.df_clean is None or not hasattr(self, 'selected_features') or not self.selected_features:
            st.error("数据或特征未准备好，请先完成数据预处理和特征选择！")
            return
            
        X = self.df_clean[self.selected_features].copy()
        y = self.df_clean[target_col].copy()
        if categorical_features:
            st.write("检测到分类特征，将进行编码处理")
            le = LabelEncoder()
            for col in categorical_features:
                if col in X.columns:
                    X[col] = le.fit_transform(X[col].astype(str))
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        st.write(f"特征形状: {X.shape}")
        st.write(f"目标变量形状: {y.shape}")
        if y.dtype in ['int64', 'float64'] and len(y.unique()) > 10:
            self.problem_type = "回归"
            self.models = self._get_regression_models()
        else:
            self.problem_type = "分类"
            # 确保分类标签从0开始（对XGBoost等模型很重要）
            if y.min() != 0 and XGBClassifier is not None:
                st.info(f"注意：分类标签已从 {y.min()} 调整为从0开始，以兼容XGBoost等模型")
                y = y - y.min()
            self.models = self._get_classification_models()
        st.write(f"检测到的问题类型: {self.problem_type}")
        self.X_scaled = X_scaled
        self.y = y
    
    def _get_regression_models(self):
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
        if LGBMRegressor is not None:
            models["LightGBM回归"] = LGBMRegressor()
        return models
    
    def _get_classification_models(self):
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
        if LGBMClassifier is not None:
            models["LightGBM分类"] = LGBMClassifier()
        return models
    
    def _render_model_training(self):
        st.subheader("3. 多模型训练与对比")
        model_names = list(self.models.keys())
        self.selected_models = st.multiselect("选择要训练的模型（可多选）", model_names, default=model_names[:2])
        test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
        if st.button("开始训练所有模型"):
            with st.spinner("正在训练所有模型..."):
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    self.X_scaled, self.y, test_size=test_size, random_state=42
                )
                results = []
                best_score = None
                best_model = None
                best_model_name = None
                for name in self.selected_models:
                    model = self.models[name]
                    try:
                        model.fit(self.X_train, self.y_train)
                        y_pred = model.predict(self.X_test)
                        if self.problem_type == "回归":
                            r2 = r2_score(self.y_test, y_pred)
                            mse = mean_squared_error(self.y_test, y_pred)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(self.y_test, y_pred)
                            # 修复MAPE计算，确保使用数值而非列表
                            y_test_values = np.array(self.y_test)
                            mape = np.mean(np.abs((y_test_values - y_pred) / (y_test_values + 1e-8))) * 100
                            results.append({
                                "模型": name,
                                "R²分数": f"{r2:.4f}",
                                "均方误差(MSE)": f"{mse:.4f}",
                                "均方根误差(RMSE)": f"{rmse:.4f}",
                                "平均绝对误差(MAE)": f"{mae:.4f}",
                                "平均绝对百分比误差(MAPE%)": f"{mape:.2f}"
                            })
                            if best_score is None or r2 > best_score:
                                best_score = r2
                                best_model = model
                                best_model_name = name
                        else:
                            acc = accuracy_score(self.y_test, y_pred)
                            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division='warn')
                            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division='warn')
                            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division='warn')
                            # AUC只对二分类且有predict_proba支持的模型有效
                            auc = None
                            if hasattr(model, 'predict_proba') and len(np.unique(self.y_test)) == 2:
                                try:
                                    auc = roc_auc_score(self.y_test, model.predict_proba(self.X_test)[:, 1])
                                except Exception:
                                    auc = None
                            results.append({
                                "模型": name,
                                "准确率": f"{acc:.4f}",
                                "F1分数": f"{f1:.4f}",
                                "精确率": f"{precision:.4f}",
                                "召回率": f"{recall:.4f}",
                                "AUC": f"{auc:.4f}" if auc is not None else "-"
                            })
                            if best_score is None or acc > best_score:
                                best_score = acc
                                best_model = model
                                best_model_name = name
                    except Exception as e:
                        results.append({"模型": name, "错误": str(e)})
                st.write("## 各模型表现对比：")
                st.dataframe(pd.DataFrame(results))
                if best_model is not None:
                    st.success(f"推荐最佳模型：{best_model_name}")
                    
                    # 添加特征重要性可视化
                    if hasattr(best_model, 'feature_importances_'):
                        st.subheader("特征重要性")
                        importances = best_model.feature_importances_
                        indices = np.argsort(importances)[::-1]
                        feature_names = np.array(self.selected_features)[indices]
                        importances = importances[indices]
                        
                        fig = px.bar(
                            x=importances, 
                            y=feature_names,
                            orientation='h',
                            labels={'x': '重要性', 'y': '特征'},
                            title='特征重要性排序'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if st.button(f"设为当前模型: {best_model_name}"):
                        self._set_best_model(best_model, best_model_name)
                        st.success(f"已将 {best_model_name} 设为当前模型，可用于后续预测和评估！")
    
    def _set_best_model(self, model, model_name):
        st.session_state['model'] = model
        st.session_state['scaler'] = self.scaler
        st.session_state['X_test'] = self.X_test
        st.session_state['y_test'] = self.y_test
        st.session_state['y_pred'] = model.predict(self.X_test)
        st.session_state['problem_type'] = self.problem_type
        st.session_state['selected_features'] = self.selected_features
        st.session_state['df_clean'] = self.df_clean
        st.session_state['best_model_name'] = model_name 