import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import itertools
import joblib

df = pd.read_csv('Crop_recommendation.csv')

# 分离标签与特征值
x = df.drop(['label'], axis=1) # 特征值
Y = df['label'] # 标签
labels = df['label'].tolist()
encode = preprocessing.LabelEncoder()
y = encode.fit_transform(Y) # 标签编码
print(y)

# 分离测试集以及数据集
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=10)

# 构建随机森林模型
model = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=5)
model.fit(x_train, y_train)

# 特征重要性分析
model.feature_importances_
feature_names = x_test.columns
feature_importances = model.feature_importances_
indices = np.argsort(feature_importances)[::-1]
plt.figure()
plt.title("Feature Importance")
plt.bar(range(len(feature_importances)), feature_importances[indices], color='b')
plt.xticks(range(len(feature_importances)), np.array(feature_names)[indices], color='b', rotation=90)
plt.savefig('my_plot.png')

# 预测测试集
y_pred = model.predict(x_test) # 定性数据
y_pred_quant = model.predict_proba(x_test) # 定量数据

# 混淆矩阵
confusion_matrix_model = confusion_matrix(y_test, y_pred)

# 混淆矩阵热力图绘制函数
def cnf_matrix_plotter(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues) # 更换颜色映射为蓝色
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # 标准化数字格式，四舍五入到整数
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    plt.savefig('Confusion_matrix_heat_map.png')

# 调用方法绘制混淆矩阵热力图
cnf_matrix_plotter(confusion_matrix_model, ['rice','maize','chickpea','kidneybeans','pigeonpeas','mothbeans','mungbean','blackgram','lentil', 'pomegranate','banana','mango','grapes','watermelon','muskmelon','apple','orange','papaya','coconut','cotton','jute','coffee',''])


# 将模型保存为.pkl 文件
joblib.dump(model, 'random_forest_model.pkl')

# 创建并拟合标签编码器
encoder = preprocessing.LabelEncoder()
encoder.fit(labels)  # labels 是你的标签数据
# 将标签编码器保存为.pkl 文件
joblib.dump(encoder, 'label_encoder.pkl')