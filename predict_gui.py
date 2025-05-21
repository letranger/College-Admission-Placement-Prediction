import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import pandas as pd

class PredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("大學入學預測系統")
        self.root.geometry("800x600")
        
        # 載入模型和編碼器
        try:
            self.school_model = joblib.load('school_model.joblib')
            self.method_model = joblib.load('method_model.joblib')
            self.school_encoder = joblib.load('school_encoder.joblib')
            self.method_encoder = joblib.load('method_encoder.joblib')
        except:
            messagebox.showerror("錯誤", "找不到模型檔案，請先運行 main.py 訓練模型")
            root.destroy()
            return

        # 創建輸入框架
        input_frame = ttk.LabelFrame(root, text="輸入學生成績", padding="10")
        input_frame.pack(fill="x", padx=10, pady=5)

        # 成績輸入欄位
        self.grade_entries = {}
        grades = [
            ('10ASub1', '10年級上學期科目1'),
            ('10ASub2', '10年級上學期科目2'),
            ('10ASub3', '10年級上學期科目3'),
            ('10BSub1', '10年級下學期科目1'),
            ('10BSub2', '10年級下學期科目2'),
            ('10BSub3', '10年級下學期科目3'),
            ('11ASub1', '11年級上學期科目1'),
            ('11ASub2', '11年級上學期科目2'),
            ('11ASub3', '11年級上學期科目3'),
            ('12ASub1', '12年級上學期科目1'),
            ('12ASub2', '12年級上學期科目2'),
            ('12ASub3', '12年級上學期科目3')
        ]

        # 創建網格布局
        for i, (field, label) in enumerate(grades):
            row = i // 3
            col = i % 3 * 2
            ttk.Label(input_frame, text=label).grid(row=row, column=col, padx=5, pady=2)
            entry = ttk.Entry(input_frame, width=10)
            entry.grid(row=row, column=col+1, padx=5, pady=2)
            self.grade_entries[field] = entry

        # 預測按鈕
        predict_button = ttk.Button(root, text="預測", command=self.predict)
        predict_button.pack(pady=10)

        # 結果顯示框架
        result_frame = ttk.LabelFrame(root, text="預測結果", padding="10")
        result_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # 學校預測結果
        school_frame = ttk.Frame(result_frame)
        school_frame.pack(fill="x", pady=5)
        ttk.Label(school_frame, text="預測學校：").pack(side="left")
        self.school_result = ttk.Label(school_frame, text="")
        self.school_result.pack(side="left")

        # 入學管道預測結果
        method_frame = ttk.Frame(result_frame)
        method_frame.pack(fill="x", pady=5)
        ttk.Label(method_frame, text="預測入學管道：").pack(side="left")
        self.method_result = ttk.Label(method_frame, text="")
        self.method_result.pack(side="left")

        # 機率顯示
        prob_frame = ttk.LabelFrame(result_frame, text="預測機率", padding="10")
        prob_frame.pack(fill="both", expand=True, pady=5)
        
        # 學校機率
        self.school_prob_text = tk.Text(prob_frame, height=5, width=40)
        self.school_prob_text.pack(side="left", padx=5)
        
        # 入學管道機率
        self.method_prob_text = tk.Text(prob_frame, height=5, width=40)
        self.method_prob_text.pack(side="left", padx=5)

    def predict(self):
        try:
            # 收集輸入的成績
            input_data = {}
            for field, entry in self.grade_entries.items():
                value = entry.get()
                if not value:
                    messagebox.showerror("錯誤", f"請輸入{field}的成績")
                    return
                try:
                    input_data[field] = float(value)
                except ValueError:
                    messagebox.showerror("錯誤", f"{field}的成績必須是數字")
                    return

            # 轉換為DataFrame
            input_df = pd.DataFrame([input_data])

            # 預測學校
            school_probs = self.school_model.predict_proba(input_df)[0]
            school_pred = self.school_model.predict(input_df)[0]
            school_name = self.school_encoder.inverse_transform([school_pred])[0]

            # 預測入學管道
            method_probs = self.method_model.predict_proba(input_df)[0]
            method_pred = self.method_model.predict(input_df)[0]
            method_name = self.method_encoder.inverse_transform([method_pred])[0]

            # 更新結果顯示
            self.school_result.config(text=school_name)
            self.method_result.config(text=method_name)

            # 顯示機率
            self.school_prob_text.delete(1.0, tk.END)
            self.method_prob_text.delete(1.0, tk.END)

            # 學校機率
            school_probs_text = "學校預測機率：\n"
            for i, prob in enumerate(school_probs):
                school_name = self.school_encoder.inverse_transform([i])[0]
                school_probs_text += f"{school_name}: {prob:.2%}\n"
            self.school_prob_text.insert(1.0, school_probs_text)

            # 入學管道機率
            method_probs_text = "入學管道預測機率：\n"
            for i, prob in enumerate(method_probs):
                method_name = self.method_encoder.inverse_transform([i])[0]
                method_probs_text += f"{method_name}: {prob:.2%}\n"
            self.method_prob_text.insert(1.0, method_probs_text)

        except Exception as e:
            messagebox.showerror("錯誤", f"預測過程發生錯誤：{str(e)}")

def main():
    root = tk.Tk()
    app = PredictionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 