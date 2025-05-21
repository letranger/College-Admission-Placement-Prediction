import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 設置頁面配置
st.set_page_config(
    page_title="[學測卜聖卦]學測落點預測系統",
    page_icon="🎓",
    layout="wide"
)

# 添加標題和說明
st.title("🎓 [學測卜聖卦]學測落點預測系統")
st.markdown("""
這個系統可以根據學生的學科成績，預測可能錄取的學校和入學管道。
請在下方輸入學生的各科成績，系統會自動進行預測。本系統以台南一中學生成績做為模型訓練依據，故預測結果僅供參考。
""")

# 載入模型和編碼器
@st.cache_resource
def load_models():
    try:
        school_model = joblib.load('models/school_model.joblib')
        method_model = joblib.load('models/method_model.joblib')
        school_encoder = joblib.load('models/school_encoder.joblib')
        method_encoder = joblib.load('models/method_encoder.joblib')
        return school_model, method_model, school_encoder, method_encoder
    except:
        st.error("找不到模型檔案，請先運行 main.py 訓練模型")
        return None, None, None, None

school_model, method_model, school_encoder, method_encoder = load_models()

if school_model is not None:
    # 創建兩列布局
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 輸入學生成績")
        
        # 創建成績輸入欄位
        grades = {
            '高一上學期': {
                '10ASub1': '英文',
                '10ASub2': '數學',
                '10ASub3': '國文'
            },
            '高一下學期': {
                '10BSub1': '英文',
                '10BSub2': '數學',
                '10BSub3': '國文'
            },
            '高二上學期': {
                '11ASub1': '英文',
                '11ASub2': '數學',
                '11ASub3': '國文'
            },
            '高二下學期': {
                '11BSub1': '英文',
                '11BSub2': '數學',
                '11BSub3': '國文'
            }
        }

        input_data = {}
        for semester, subjects in grades.items():
            st.markdown(f"### {semester}")
            cols = st.columns(3)
            for i, (field, subject) in enumerate(subjects.items()):
                with cols[i]:
                    input_data[field] = st.number_input(
                        subject,
                        min_value=0,
                        max_value=100,
                        step=1,
                        key=field
                    )

    with col2:
        st.subheader("🎯 預測結果")
        
        if st.button("開始預測", type="primary"):
            # 轉換為DataFrame
            input_df = pd.DataFrame([input_data])
            
            # 預測學校
            school_probs = school_model.predict_proba(input_df)[0]
            school_pred = school_model.predict(input_df)[0]
            school_name = school_encoder.inverse_transform([school_pred])[0]
            
            # 預測入學管道
            method_probs = method_model.predict_proba(input_df)[0]
            method_pred = method_model.predict(input_df)[0]
            method_name = method_encoder.inverse_transform([method_pred])[0]
            
            # 顯示主要預測結果
            st.markdown("### 主要預測")
            st.markdown(f"**預測學校：** {school_name}")
            st.markdown(f"**預測入學管道：** {method_name}")
            
            # 顯示詳細機率
            st.markdown("### 詳細預測機率")
            
            # 學校機率
            st.markdown("#### 學校錄取機率")
            school_probs_df = pd.DataFrame({
                '學校': school_encoder.inverse_transform(range(len(school_probs))),
                '機率': school_probs
            }).sort_values('機率', ascending=False)
            
            # 使用條形圖顯示學校機率
            st.bar_chart(school_probs_df.set_index('學校')['機率'])
            
            # 入學管道機率
            st.markdown("#### 入學管道機率")
            method_probs_df = pd.DataFrame({
                '入學管道': method_encoder.inverse_transform(range(len(method_probs))),
                '機率': method_probs
            }).sort_values('機率', ascending=False)
            
            # 使用條形圖顯示入學管道機率
            st.bar_chart(method_probs_df.set_index('入學管道')['機率'])

# 添加頁尾
st.markdown("---")
st.markdown("### 📊 系統說明")
st.markdown("""
- 本系統使用機器學習模型進行預測
- 預測結果僅供參考
- 實際錄取結果可能受到多種因素影響
""") 