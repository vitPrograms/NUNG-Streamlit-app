import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

df = pd.read_csv('dataset/ObesityDataSet_raw_and_data_sinthetic.csv')
print(df.head())
print(df.info())

X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

print(f"Категоріальні стовпці: {categorical_cols.tolist()}")
print(f"Числові стовпці: {numerical_cols.tolist()}")

le = LabelEncoder()
y_encoded = le.fit_transform(y)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

X_processed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(le.classes_), activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
# ===== ВІЗУАЛІЗАЦІЇ: ФУНКЦІЇ =====

try:
    plt.figure(figsize=(10, 4))
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get('loss', []), label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get('accuracy', []), label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"Графік кривих навчання збережено у {out_path}")
except Exception as e:
    print('Не вдалося намалювати криві навчання:', e)

try:
    cm = confusion_matrix(y_true, y_pred)
    print("\nКласифікаційний звіт:\n", classification_report(y_true, y_pred, target_names=class_names))
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                      color='white' if cm[i, j] > thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"Матрицю змішування збережено у {out_path}")
except Exception as e:
    print('Не вдалося побудувати матрицю змішування:', e)


loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f'Точність на тестовому наборі: {accuracy:.4f}')
print(f'Втрати на тестовому наборі: {loss:.4f}')

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nПрогнози на тестовому наборі (перші 10):")
for i in range(10):
    print(f"Прогноз: {le.inverse_transform([y_pred[i]])[0]}, Фактично: {le.inverse_transform([y_test[i]])[0]}")


import joblib
model.save('obesity_model.h5')
joblib.dump(preprocessor, 'scaler.pkl')  # Замість StandardScaler окремо
# Після preprocessor.fit_transform(X)
columns = preprocessor.get_feature_names_out()  # Отримуємо правильні назви стовпців
joblib.dump(columns, 'columns.pkl')
