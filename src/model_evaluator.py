"""
模型评估模块
负责模型性能评估和新数据预测
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class ModelEvaluator:
    """模型评估器类"""
    
    def __init__(self):
        """初始化模型评估器"""
        pass
    
    def render(self):
        """渲染模型评估界面"""
        st.header("模型评估与预测")
        
        if 'model' in st.session_state:
            self._render_model_evaluation()
            self._render_prediction_interface()
        else:
            st.info("请先在'模型预测'标签页中训练模型。")
    
    def _render_model_evaluation(self):
        """渲染模型性能评估"""
        st.subheader("1. 模型性能评估")
        
        # 显示测试集预测结果
        if st.session_state['problem_type'] == "回归":
            self._render_regression_evaluation()
        else:
            self._render_classification_evaluation()
    
    def _render_regression_evaluation(self):
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
    
    def _render_classification_evaluation(self):
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
        from sklearn.metrics import classification_report
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
            from sklearn.metrics import roc_curve, auc
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
    
    def _render_prediction_interface(self):
        """渲染预测界面"""
        st.subheader("2. 新数据预测")
        
        # 创建预测输入表单
        st.write("输入新数据进行预测：")
        
        input_data = {}
        df_clean = st.session_state.get('df_clean')
        
        if df_clean is not None:
            for feature in st.session_state['selected_features']:
                # 获取该特征的数据范围用于设置默认值
                feature_data = df_clean[feature]
                default_value = float(feature_data.mean())
                min_val = float(feature_data.min())
                max_val = float(feature_data.max())
                
                input_data[feature] = st.number_input(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_value,
                    step=(max_val - min_val) / 100
                )
            
            if st.button("进行预测"):
                self._make_prediction(input_data)
    
    def _make_prediction(self, input_data: dict):
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
    
    def get_model_summary(self) -> dict:
        """获取模型摘要信息"""
        if 'model' not in st.session_state:
            return {}
        
        summary = {
            'problem_type': st.session_state.get('problem_type'),
            'model_name': type(st.session_state['model']).__name__,
            'features': st.session_state.get('selected_features', []),
            'test_size': len(st.session_state.get('X_test', [])),
            'train_size': len(st.session_state.get('y_test', [])) - len(st.session_state.get('X_test', []))
        }
        
        return summary 