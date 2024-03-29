import numpy as np
import pandas as pd
import joblib

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    # 加载模型
    model = joblib.load('random_forest_model.pkl')
    # 加载标签编码器
    encoder = joblib.load('label_encoder.pkl')

    # 构建输入数据
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    # 进行预测
    prediction = model.predict(data)
    # 获取预测结果的标签
    predicted_label = encoder.inverse_transform(prediction)[0]

    return predicted_label

def main():
    print("请输入以下参数以预测作物类型:")
    N = float(input("土壤中的氮含量: "))
    P = float(input("土壤中的磷含量: "))
    K = float(input("土壤中的钾含量: "))
    temperature = float(input("气温（摄氏度）: "))
    humidity = float(input("湿度（百分比）: "))
    ph = float(input("土壤的pH值: "))
    rainfall = float(input("降雨量（毫米）: "))

    predicted_label = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
    print(f"预测的作物类型为: {predicted_label}")

if __name__ == "__main__":
    main()
