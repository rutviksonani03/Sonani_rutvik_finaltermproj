# CS634 Final Project - Diabetes Prediction


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, brier_score_loss
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings("ignore")


# 1. Load Dataset


print("Loading diabetes dataset...")
diabetes = pd.read_csv('diabetes.csv')

print("\nFirst 5 rows:")
print(diabetes.head())


# 2. Impute Zeros with Median


def impute_missing_zeros(data):
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_fix:
        median_val = data[col][data[col] != 0].median()
        data[col] = data[col].replace(0, median_val)
    return data

diabetes = impute_missing_zeros(diabetes)
print("\nZeros replaced with median values.")


# 3. Data Exploration


print("\n" + "="*60)
print("DATA EXPLORATION")
print("="*60)

plt.figure(figsize=(7,5))
sns.countplot(x='Outcome', data=diabetes, palette='Set2')
plt.title('Diabetes Outcome Distribution')
plt.xlabel('Outcome (0: No, 1: Yes)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('outcome_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

pos = diabetes['Outcome'].value_counts()[1]
neg = diabetes['Outcome'].value_counts()[0]
total = len(diabetes)
print(f"Positive (Diabetes): {pos} ({pos/total*100:.1f}%)")
print(f"Negative (No Diabetes): {neg} ({neg/total*100:.1f}%)")

plt.figure(figsize=(10,8))
corr_matrix = diabetes.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()


# 4. Features and Target


X = diabetes.drop('Outcome', axis=1)
y = diabetes['Outcome']


# 5. Custom Metrics Function


def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    p = tp + fn
    n = tn + fp

    tpr = tp / p if p > 0 else 0
    tnr = tn / n if n > 0 else 0
    fpr = fp / n if n > 0 else 0
    fnr = fn / p if p > 0 else 0

    acc = (tp + tn) / (p + n)
    bal_acc = (tpr + tnr) / 2
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tpr
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    err = 1 - acc

    tss = tpr - fpr
    hss_num = 2 * (tp * tn - fp * fn)
    hss_den = (tp + fn)*(fn + tn) + (tp + fp)*(fp + tn)
    hss = hss_num / hss_den if hss_den > 0 else 0

    auc = roc_auc_score(y_true, y_prob)
    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_prob)
    bs = brier_score_loss(y_true, y_prob)
    ref_bs = y_true.mean() * (1 - y_true.mean())
    bss = 1 - (bs / ref_bs) if ref_bs > 0 else 0

    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'P': p, 'N': n,
        'TPR': tpr, 'TNR': tnr, 'FPR': fpr, 'FNR': fnr,
        'Accuracy': acc, 'Balanced Accuracy': bal_acc,
        'Precision': prec, 'Recall': rec, 'F1': f1, 'Error Rate': err,
        'TSS': tss, 'HSS': hss, 'AUC': auc, 'BS': bs, 'BSS': bss,
        'fpr_curve': fpr_curve, 'tpr_curve': tpr_curve
    }

# 6. 10-Fold Stratified CV with FULL PER-FOLD METRIC TABLES

cv_folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scaler = StandardScaler()

# Store results
rf_results = []
lstm_results = []
svm_results = []
rf_probs = []
lstm_probs = []
svm_probs = []

print("\nStarting 10-fold cross-validation...\n")

for fold, (train_idx, test_idx) in enumerate(cv_folds.split(X, y), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # === Random Forest ===
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_prob = rf_model.predict_proba(X_test_scaled)[:, 1]
    rf_metrics = compute_metrics(y_test, rf_pred, rf_prob)
    rf_results.append(rf_metrics)
    rf_probs.append((y_test.values, rf_prob))

    # === SVM ===
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    svm_pred = svm_model.predict(X_test_scaled)
    svm_prob = svm_model.predict_proba(X_test_scaled)[:, 1]
    svm_metrics = compute_metrics(y_test, svm_pred, svm_prob)
    svm_results.append(svm_metrics)
    svm_probs.append((y_test.values, svm_prob))

    # === LSTM ===
    X_train_lstm = X_train_scaled.reshape(-1, 1, X_train_scaled.shape[1])
    X_test_lstm = X_test_scaled.reshape(-1, 1, X_test_scaled.shape[1])

    lstm_model = Sequential([
        LSTM(50, input_shape=(1, X_train_scaled.shape[1])),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam')
    lstm_model.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=0)

    lstm_prob = lstm_model.predict(X_test_lstm).flatten()
    lstm_pred = (lstm_prob > 0.5).astype(int)
    lstm_metrics = compute_metrics(y_test, lstm_pred, lstm_prob)
    lstm_results.append(lstm_metrics)
    lstm_probs.append((y_test.values, lstm_prob))

    # === PRINT FULL PER-FOLD METRICS TABLE ===
    print(f"\n{'='*70}")
    print(f" FOLD {fold} DETAILED RESULTS ")
    print(f"{'='*70}")

    def print_fold_table(metrics, model_name):
        print(f"\n--- {model_name} ---")
        skip = ['fpr_curve', 'tpr_curve']
        for k, v in metrics.items():
            if k not in skip:
                print(f"  {k:20}: {v:.4f}")

    print_fold_table(rf_metrics, "Random Forest")
    print_fold_table(svm_metrics, "SVM")
    print_fold_table(lstm_metrics, "LSTM")
    print(f"{'-'*70}")

# 7. Average Metrics

def avg_metrics(model_results):
    return {k: np.mean([m[k] for m in model_results]) for k in model_results[0] if k not in ['fpr_curve', 'tpr_curve']}

rf_avg = avg_metrics(rf_results)
lstm_avg = avg_metrics(lstm_results)
svm_avg = avg_metrics(svm_results)

print("\n" + "="*70)
print("AVERAGE RESULTS (10-FOLD CV)")
print("="*70)
for model_name, avg in zip(['Random Forest', 'LSTM', 'SVM'], [rf_avg, lstm_avg, svm_avg]):
    print(f"\n--- {model_name} ---")
    for k, v in avg.items():
        print(f"  {k}: {v:.4f}")

# 8. Summary Table

summary_data = []
keys = [k for k in rf_avg.keys()]
for k in keys:
    summary_data.append({
        'Metric': k,
        'Random Forest': round(rf_avg[k], 4),
        'LSTM': round(lstm_avg[k], 4),
        'SVM': round(svm_avg[k], 4)
    })

summary_df = pd.DataFrame(summary_data)
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(summary_df.to_string(index=False))

# 9. Average ROC Curves

def plot_roc(probs, label):
    all_y = np.concatenate([y for y, _ in probs])
    all_p = np.concatenate([p for _, p in probs])
    fpr, tpr, _ = roc_curve(all_y, all_p)
    auc = roc_auc_score(all_y, all_p)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.3f})')

plt.figure(figsize=(9,7))
plot_roc(rf_probs, 'Random Forest')
plot_roc(lstm_probs, 'LSTM')
plot_roc(svm_probs, 'SVM')
plt.plot([0,1], [0,1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Average ROC Curves (10-Fold CV)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n Plots saved.")