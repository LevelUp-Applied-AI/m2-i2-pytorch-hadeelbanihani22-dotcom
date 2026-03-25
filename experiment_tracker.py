# استيراد المكتبات (Libraries)
import pandas as pd                  # للتعامل مع البيانات (DataFrames)
import numpy as np                   # للعمليات الحسابية
import torch                         # مكتبة PyTorch
import torch.nn as nn                # nn = Neural Network (طبقات الموديل)
import itertools                     # لتوليد جميع التركيبات (Cartesian Product)
import json                          # لحفظ النتائج في ملف
import time                          # لحساب وقت التدريب
import matplotlib.pyplot as plt      # للرسم (Visualization)


# ─── تعريف الموديل ────────────────────────────────────────────────────────

class HousingModel(nn.Module):  # تعريف موديل الشبكة العصبية
    def __init__(self, input_size, hidden_size):
        super().__init__()  # تهيئة الكلاس الأساسي في PyTorch
        
        self.layer1 = nn.Linear(input_size, hidden_size)  # طبقة: 5 → hidden_size
        self.relu = nn.ReLU()                             # دالة تنشيط (تحذف القيم السالبة)
        self.layer2 = nn.Linear(hidden_size, 1)           # طبقة: hidden_size → 1 (السعر)

    def forward(self, x):  # كيف البيانات تمشي داخل الموديل
        x = self.layer1(x)   # تمرير عبر الطبقة الأولى
        x = self.relu(x)     # تطبيق ReLU
        x = self.layer2(x)   # إخراج النتيجة النهائية
        return x


# ─── دالة التدريب ───────────────────────────────────────────────────────

def train_model(X_train, y_train, X_test, y_test, config):

    # إنشاء موديل جديد لكل تجربة
    model = HousingModel(input_size=5, hidden_size=config["hidden_size"])
    
    criterion = nn.MSELoss()  # دالة الخطأ (Loss Function)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])  # Optimizer

    start_time = time.time()  # بداية حساب الوقت

    # ─── تدريب الموديل ───────────────────────────────────────────────
    for epoch in range(config["epochs"]):
        preds = model(X_train)             # Forward pass (توقعات)
        loss = criterion(preds, y_train)   # حساب الخطأ

        optimizer.zero_grad()  # تصفير القيم القديمة للـ gradients
        loss.backward()        # Backpropagation (حساب المشتقات)
        optimizer.step()       # تحديث الأوزان

    train_loss = loss.item()   # آخر قيمة loss بعد التدريب

    # ─── التقييم على test set ───────────────────────────────────────
    with torch.no_grad():  # بدون حساب gradients (أسرع)
        test_preds_tensor = model(X_test)                  # توقعات test
        test_loss = criterion(test_preds_tensor, y_test).item()  # خطأ test

        test_preds = test_preds_tensor.numpy().flatten()  # تحويل لنوع numpy
        test_actual = y_test.numpy().flatten()            # القيم الحقيقية

    # حساب MAE (Mean Absolute Error)
    mae = np.mean(np.abs(test_actual - test_preds))

    # حساب R² (Coefficient of Determination)
    ss_res = np.sum((test_actual - test_preds) ** 2)
    ss_tot = np.sum((test_actual - np.mean(test_actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    end_time = time.time()  # نهاية الوقت

    # إرجاع النتائج
    return {
        "config": config,
        "train_loss": float(train_loss),
        "test_loss": float(test_loss),
        "mae": float(mae),
        "r2": float(r2),
        "time": round(end_time - start_time, 2)
    }


# ─── الدالة الرئيسية ─────────────────────────────────────────────────────

def main():

    # ── تحميل البيانات ──
    df = pd.read_csv("data/housing.csv")  # قراءة ملف CSV
    print("Data shape:", df.shape)        # طباعة عدد الصفوف والأعمدة

    # ── تحديد الخصائص والهدف ──
    feature_cols = ['area_sqm', 'bedrooms', 'floor', 'age_years', 'distance_to_center_km']
    
    X = df[feature_cols]       # المدخلات (Features)
    y = df[['price_jod']]      # الهدف (السعر)

    # ── تحويل إلى Tensor ──
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    # ── تقسيم البيانات Train/Test ──
    torch.manual_seed(42)  # تثبيت العشوائية (Reproducibility)

    indices = torch.randperm(len(X_tensor))  # ترتيب عشوائي
    X_shuffled = X_tensor[indices]
    y_shuffled = y_tensor[indices]

    split = int(0.8 * len(X_tensor))  # 80% تدريب

    X_train = X_shuffled[:split]
    X_test = X_shuffled[split:]
    y_train = y_shuffled[:split]
    y_test = y_shuffled[split:]

    # ── Standardization (بدون Data Leakage) ──
    X_mean = X_train.mean(dim=0)  # متوسط train فقط
    X_std = X_train.std(dim=0)    # الانحراف المعياري

    X_train = (X_train - X_mean) / X_std  # توحيد train
    X_test = (X_test - X_mean) / X_std    # توحيد test بنفس القيم

    # ── Hyperparameter Grid ──
    learning_rates = [0.001, 0.01, 0.05]
    hidden_sizes = [16, 32, 64]
    epochs_list = [50, 100, 150, 200]  # 36 تجربة

    experiments = []

    print("\nRunning experiments...\n")

    # ── Loop للتجارب ──
    for lr, hs, ep in itertools.product(learning_rates, hidden_sizes, epochs_list):

        config = {
            "learning_rate": lr,
            "hidden_size": hs,
            "epochs": ep
        }

        result = train_model(X_train, y_train, X_test, y_test, config)
        experiments.append(result)

        print(f"Done: {config} | MAE: {result['mae']:.2f}")

    # ── حفظ النتائج ──
    with open("experiments.json", "w") as f:
        json.dump(experiments, f, indent=4)

    print("\nSaved experiments.json")

    # ── ترتيب النتائج (Leaderboard) ──
    sorted_exp = sorted(experiments, key=lambda x: x["mae"])

    print("\nTop 10 Models:")
    print("Rank | LR | Hidden | Epochs | MAE | R2 | Time")

    for i, exp in enumerate(sorted_exp[:10]):
        c = exp["config"]
        print(f"{i+1:2d} | {c['learning_rate']} | {c['hidden_size']} | {c['epochs']} | {exp['mae']:.2f} | {exp['r2']:.2f} | {exp['time']}s")

    # ── رسم النتائج ──
    lrs = [e["config"]["learning_rate"] for e in experiments]
    maes = [e["mae"] for e in experiments]

    plt.figure()  # إنشاء رسم جديد
    plt.scatter(lrs, maes)  # رسم نقاط
    plt.xlabel("Learning Rate")  # اسم المحور X
    plt.ylabel("MAE")            # اسم المحور Y
    plt.title("MAE vs Learning Rate")  # عنوان الرسم

    plt.savefig("experiment_summary.png")  # حفظ الصورة

    print("\nSaved experiment_summary.png")


# ── تشغيل البرنامج ──
if __name__ == "__main__":
    main()