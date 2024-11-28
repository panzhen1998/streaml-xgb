#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
data = pd.read_csv('C:\Users\27854\Desktop\test.csv')
df = pd.DataFrame(data)


# In[2]:


# 划分特征和目标变量
X = df.drop(['target'], axis=1)
y = df['target']
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,                                                     
                                                    random_state=42, stratify=df['target'])
df.head()


# In[3]:


import xgboost as xgb
from sklearn.model_selection import GridSearchCV


# In[4]:


# XGBoost模型参数
params_xgb = {    'learning_rate': 0.02,            # 学习率，控制每一步的步长，用于防止过拟合。典型值范围：0.01 - 0.1    
              'booster': 'gbtree',              # 提升方法，这里使用梯度提升树（Gradient Boosting Tree）    
              'objective': 'binary:logistic',   # 损失函数，这里使用逻辑回归，用于二分类任务    
              'max_leaves': 127,                # 每棵树的叶子节点数量，控制模型复杂度。较大值可以提高模型复杂度但可能导致过拟合    
              'verbosity': 1,                   # 控制 XGBoost 输出信息的详细程度，0表示无输出，1表示输出进度信息    
              'seed': 42,                       # 随机种子，用于重现模型的结果    
              'nthread': -1,                    # 并行运算的线程数量，-1表示使用所有可用的CPU核心    
              'colsample_bytree': 0.6,          # 每棵树随机选择的特征比例，用于增加模型的泛化能力    
              'subsample': 0.7,                 # 每次迭代时随机选择的样本比例，用于增加模型的泛化能力    
              'eval_metric': 'logloss'          # 评价指标，这里使用对数损失（logloss）
             }


# In[5]:


# 初始化XGBoost分类模型
model_xgb = xgb.XGBClassifier(**params_xgb)


# In[6]:


# 定义参数网格，用于网格搜索
param_grid = {    'n_estimators': [100, 200, 300, 400, 500],  # 树的数量    
              'max_depth': [3, 4, 5, 6, 7],               # 树的深度    
              'learning_rate': [0.01, 0.02, 0.05, 0.1],   # 学习率
             }


# In[7]:


# 使用GridSearchCV进行网格搜索和k折交叉验证
grid_search = GridSearchCV(    estimator=model_xgb,    
                           param_grid=param_grid,    
                           scoring='neg_log_loss',  # 评价指标为负对数损失    
                           cv=5,                    # 5折交叉验证    
                           n_jobs=-1,               # 并行计算    
                           verbose=1                # 输出详细进度信息
                          )


# In[8]:


# 训练模型
grid_search.fit(X_train, y_train)
# 输出最优参数
print("Best parameters found: ", grid_search.best_params_)
print("Best Log Loss score: ", -grid_search.best_score_)
# 使用最优参数训练模型
best_model = grid_search.best_estimator_


# In[9]:


from sklearn.metrics import classification_report
# 预测测试集
y_pred = best_model.predict(X_test)
# 输出模型报告， 查看评价指标
print(classification_report(y_test, y_pred))


# In[10]:


from sklearn.metrics import roc_curve, auc
# 预测概率
y_score = best_model.predict_proba(X_test)[:, 1]
# 计算ROC曲线
fpr_logistic, tpr_logistic, _ = roc_curve(y_test, y_score)
roc_auc_logistic = auc(fpr_logistic, tpr_logistic)
# 绘制ROC曲线
plt.figure()
plt.plot(fpr_logistic, tpr_logistic, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_logistic)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[11]:


import joblib
# 保存模型
joblib.dump(best_model , 'XGBoost.pkl')


# In[12]:


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
# Load the model
model = joblib.load('XGBoost.pkl')
# Define feature options
cN_options = {    
    0: 'No lymph node metastasis (0)',    
    1: 'Lymph node metastasis (1)'    
}
# Define feature names
feature_names = [   
    "TMRL", "cN", "distance", "size"
]
# Streamlit user interface
st.title("LARC Disease Predictor")
# age: numerical input
TMRL = st.number_input("TMRL:", min_value=-70.0, max_value=3.0, value=1.0)
size = st.number_input("Tumor size", min_value=0.1, max_value=10.0, value=0.1)
distance = st.number_input("The distance from anus", min_value=0.1, max_value=16.0, value=0.1)
cN = st.selectbox("cN (0=No lymph node metastasis, 1=Lymph node metastasis):", options=[0, 1], format_func=lambda x: 'No lymph node metastasis (0)' if x == 0 else 'Lymph node metastasis (1)')
# Process inputs and make predictions
feature_values = [TMRL, cN, distance, size]
features = np.array([feature_values])
if st.button("Predict"):    
    # Predict class and probabilities    
    predicted_class = model.predict(features)[0]    
    predicted_proba = model.predict_proba(features)[0]
    # Display prediction results    
    st.write(f"**Predicted Class:** {predicted_class}")    
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    # Generate advice based on prediction results    
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:        
        advice = (            
            f"According to our model, you have a high risk of heart disease. "            
            f"The model predicts that your probability of having heart disease is {probability:.1f}%. "            
            "While this is just an estimate, it suggests that you may be at significant risk. "            
            "I recommend that you consult a cardiologist as soon as possible for further evaluation and "            
            "to ensure you receive an accurate diagnosis and necessary treatment."        
        ) 
    else:        
        advice = (            
            f"According to our model, you have a low risk of heart disease. "            
            f"The model predicts that your probability of not having heart disease is {probability:.1f}%. "            
            "However, maintaining a healthy lifestyle is still very important. "            
            "I recommend regular check-ups to monitor your heart health, "            
            "and to seek medical advice promptly if you experience any symptoms."        
        )
    st.write(advice)
    # Calculate SHAP values and display force plot    
    explainer = shap.TreeExplainer(model)    
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




