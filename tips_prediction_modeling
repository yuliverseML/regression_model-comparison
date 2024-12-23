import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = sns.load_dataset('tips')
df.head()
df.shape
df.info()
df.describe()

df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'day', 'time'])

X = df_encoded.drop('tip', axis=1)
y = df_encoded['tip']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    results[name] = {
        'MSE': mse,
        'R2': r2,
        'CV_R2_mean': np.mean(cv_scores),
        'CV_R2_std': np.std(cv_scores)
    }

for name, metrics in results.items():
    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

plt.figure(figsize=(12, 8))
colors = plt.cm.rainbow(np.linspace(0, 1, len(models)))
for (name, model), color in zip(models.items(), colors):
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred, alpha=0.5, color=color, label=f"{name} (R2: {results[name]['R2']:.2f}, MSE: {results[name]['MSE']:.2f})")

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Идеальная линия')
plt.xlabel('Фактические чаевые', fontsize=12)
plt.ylabel('Предсказанные чаевые', fontsize=12)
plt.title('Сравнение фактических и предсказанных чаевых для разных моделей', fontsize=14)
plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
x = np.arange(len(models))
width = 0.35

for i, metric in enumerate(['R2', 'MSE']):
    values = [results[name][metric] for name in models]
    plt.bar(x + i*width, values, width, label=metric, color='blue' if metric=='R2' else 'red')

plt.xlabel('Модели', fontsize=12)
plt.ylabel('Значение метрики', fontsize=12)
plt.title('Сравнение метрик для разных моделей', fontsize=14)
plt.xticks(x + width/2, models.keys(), rotation=45, ha='right')
plt.legend(fontsize=10)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("\nРезультаты моделей:")
print("-" * 80)
print(f"{'Модель':<20} {'MSE':>10} {'R2':>10} {'CV R2 mean':>12} {'CV R2 std':>10}")
print("-" * 80)
for name, metrics in results.items():
    print(f"{name:<20} {metrics['MSE']:>10.4f} {metrics['R2']:>10.4f} "
          f"{metrics['CV_R2_mean']:>12.4f} {metrics['CV_R2_std']:>10.4f}")

best_model = max(results.items(), key=lambda x: x[1]['R2'] - x[1]['MSE'])[0]
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Идеальная линия')
plt.xlabel('Фактические чаевые', fontsize=12)
plt.ylabel('Предсказанные чаевые', fontsize=12)
plt.title(f'Сравнение фактических и предсказанных чаевых ({best_model_name})', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print(f"\nЛучшая модель по нашим критериям: {best_model}")
print(f"Mean Squared Error: {results[best_model_name]['MSE']:.4f}")
print(f"R-squared Score: {results[best_model_name]['R2']:.4f}")

new_total_bill = np.random.uniform(df['total_bill'].min(), df['total_bill'].max())

new_observation = pd.DataFrame({
    'total_bill': [new_total_bill],
    'size': [df['size'].mean()],  # Используем среднее значение размера группы
})

for col in X.columns:
    if col not in new_observation.columns:
        if col.startswith(('sex_', 'smoker_', 'day_', 'time_')):
            new_observation[col] = 0
        else:
            new_observation[col] = X[col].mean()

new_observation = new_observation.reindex(columns=X.columns, fill_value=0)

new_observation_scaled = scaler.transform(new_observation)

predicted_tip = best_model.predict(new_observation_scaled)[0]

print(f"\nПредсказание чаевых для счета в ${new_total_bill:.2f}: ${predicted_tip:.2f}")

np.random.seed(42)
new_feature = np.random.randn(len(X))
X_with_new_feature = np.column_stack((X, new_feature))
X_scaled_with_new = scaler.fit_transform(X_with_new_feature)
X_train_new, X_test_new, y_train, y_test = train_test_split(X_scaled_with_new, y, test_size=0.2, random_state=42)

model_new = LinearRegression()
model_new.fit(X_train_new, y_train)
y_pred_new = model_new.predict(X_test_new)
mse_new = mean_squared_error(y_test, y_pred_new)
r2_new = r2_score(y_test, y_pred_new)

print("\nМодель с новым признаком:")
print(f"Mean Squared Error: {mse_new:.4f}")
print(f"R-squared Score: {r2_new:.4f}")

print("\nСравнение с предыдущей моделью:")
print(f"MSE (предыдущая модель): {results['Linear Regression']['MSE']:.4f}")
print(f"MSE (новая модель): {mse_new:.4f}")
print(f"R2 (предыдущая модель): {results['Linear Regression']['R2']:.4f}")
print(f"R2 (новая модель): {r2_new:.4f}")

feature_importance = pd.DataFrame({
    'Feature': list(X.columns) + ['New Feature'],
    'Importance': np.abs(model_new.coef_)
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print("\nВажность признаков:")
print(feature_importance)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance['Feature'], feature_importance['Importance'])
plt.xticks(rotation=90)
plt.xlabel('Признаки')
plt.ylabel('Важность (абсолютное значение коэффициента)')
plt.title('Важность признаков в модели с новым признаком')
plt.tight_layout()
plt.show()

print("\nСравнение моделей:")
print(f"Без нового признака - MSE: {mse}, R-squared: {r2}")
print(f"С новым признаком - MSE: {mse_new}, R-squared: {r2_new}")

if r2_new > r2 or (r2_new >= r2 and mse_new < mse):
    print("Добавление нового признака улучшило модель.")
else:
    print("Добавление нового признака не улучшило модель или улучшение незначительно.")

plt.figure(figsize=(10, 6))
sns.distplot(new_feature, kde=True, hist=True)
plt.title('Распределение нового признака')
plt.xlabel('Значение нового признака')
plt.ylabel('Плотность / Частота')
plt.show()

correlation_matrix = df_encoded.assign(new_feature=new_feature).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix[['tip']].sort_values(by='tip'), annot=True, cmap='coolwarm')
plt.title('Корреляция нового признака с чаевыми')
plt.show()

def test_data_loading():
    assert isinstance(df, pd.DataFrame), "Данные не загружены в DataFrame"
    assert 'tip' in df.columns, "Колонка 'tip' отсутствует в данных"

def test_data_splitting():
    assert X_train.shape[0] + X_test.shape[0] == X_scaled.shape[0], "Неправильное разделение данных"
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0], "Неправильное разделение данных"

def test_model_training():
    assert hasattr(best_model, 'coef_'), "Модель не обучена"

def test_predictions():
    assert len(y_pred_best) == len(y_test), "Количество предсказаний не совпадает с размером тестовой выборки"

def test_model_performance():
    assert 0 <= results[best_model_name]['R2'] <= 1, "R-squared должен быть между 0 и 1"
    assert results[best_model_name]['MSE'] >= 0, "MSE должно быть неотрицательным"

test_data_loading()
test_data_splitting()
test_model_training()
test_predictions()
test_model_performance()
print("Все тесты пройдены успешно!")

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Создаем полиномиальные признаки
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Разделение данных
X_train_poly, X_test_poly, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Создаем и обучаем модель полиномиальной регрессии
poly_model = make_pipeline(StandardScaler(), LinearRegression())
poly_model.fit(X_train_poly, y_train)

# Оцениваем модель
y_pred_poly = poly_model.predict(X_test_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("\nПолиномиальная регрессия:")
print(f"Mean Squared Error: {mse_poly:.4f}")
print(f"R-squared Score: {r2_poly:.4f}")

# Сравниваем с лучшей линейной моделью
print("\nСравнение с лучшей линейной моделью:")
print(f"MSE (линейная модель): {results[best_model_name]['MSE']:.4f}")
print(f"MSE (полиномиальная модель): {mse_poly:.4f}")
print(f"R2 (линейная модель): {results[best_model_name]['R2']:.4f}")
print(f"R2 (полиномиальная модель): {r2_poly:.4f}")

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_poly, color='blue', alpha=0.5, label='Полиномиальная регрессия')
plt.scatter(y_test, best_model.predict(X_test), color='red', alpha=0.5, label='Лучшая линейная модель')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Идеальная линия')
plt.xlabel('Фактические чаевые', fontsize=12)
plt.ylabel('Предсказанные чаевые', fontsize=12)
plt.title('Сравнение полиномиальной и линейной регрессии', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

if 'new_data' in locals():
    X_new = new_data.drop('tip', axis=1)
    y_new = new_data['tip']
    X_new_scaled = scaler.transform(X_new)
    y_pred_new_data = model.predict(X_new_scaled)
    print(f"Оценка на новых данных - MSE: {mean_squared_error(y_new, y_pred_new_data)}, R²: {r2_score(y_new, y_pred_new_data)}")
else:
    print("Новые данные для тестирования не предоставлены.")

