# 學測落點預測系統

這是一個基於機器學習的學測落點預測系統，可以根據學生的學科成績預測可能錄取的學校和入學管道。

## 功能特點

- 根據學生高一、高二的學科成績進行預測
- 預測可能錄取的學校
- 預測可能的入學管道
- 提供詳細的預測機率分析
- 使用 XGBoost 機器學習模型
- 基於台南一中學生成績數據訓練

## 安裝說明

1. 克隆專案：
```bash
git clone [您的專案URL]
cd College-Admission-Placement-Prediction
```

2. 安裝依賴：
```bash
pip install -r requirements.txt
```

3. 訓練模型：
```bash
python main.py
```

4. 運行應用：
```bash
streamlit run app.py
```

## 使用說明

1. 在網頁界面輸入學生的各科成績
2. 點擊"開始預測"按鈕
3. 查看預測結果和詳細機率分析

## 技術棧

- Python
- Streamlit
- XGBoost
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## 注意事項

- 本系統的預測結果僅供參考
- 實際錄取結果可能受到多種因素影響
- 模型基於台南一中學生成績數據訓練，可能不適用於其他學校

## 授權

[您的授權信息]

## 數據說明

系統使用以下學科成績作為預測特徵：
- 高一上學期：英文、數學、國文
- 高一下學期：英文、數學、國文
- 高二上學期：英文、數學、國文
- 高二下學期：英文、數學、國文 