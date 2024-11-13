import numpy as np
import matplotlib.pyplot as plt

# Определяем целевую функцию
def f(x1, x2):
    return x1**2 - 3*x1*x2 + 10*x2**2 + 5*x1 - 3*x2

# Определяем градиент функции
def gradient(x):
    x1, x2 = x
    df_dx1 = 2*x1 - 3*x2 + 5  # Частная производная по x1
    df_dx2 = -3*x1 + 20*x2 - 3  # Частная производная по x2
    return np.array([df_dx1, df_dx2])

# Метод для нахождения оптимального шага
def optimal_step(x):
    grad = gradient(x)
    return -np.dot(grad, grad) / np.dot(grad, gradient(x + 0.001 * grad))

# Градиентный метод с оптимальным шагом
def gradient_descent(f, x0, tol=1e-6, max_iter=1000):
    x = x0
    for iteration in range(max_iter):
        grad = gradient(x)
        step_size = optimal_step(x)  # Находим оптимальный шаг
        x_new = x - step_size * grad  # Обновляем значение
        if np.linalg.norm(x_new - x) < tol:  # Проверка на сходимость
            break
        x = x_new
    return x, f(*x), iteration + 1  # Возвращаем координаты минимума, значение функции и количество итераций

# Начальная точка
x0 = np.array([2, 1])

# Запуск градиентного спуска
min_point, min_value, iterations = gradient_descent(f, x0)

# Вывод результатов
print(f"Координаты точки минимума: {min_point}")
print(f"Минимальное значение функции: {min_value}")
print(f"Количество итераций: {iterations}")

# Построение графика функции
x1 = np.linspace(-2, 4, 400)
x2 = np.linspace(-2, 4, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)

# Создание графика
plt.figure(figsize=(10, 6))
contour = plt.contour(X1, X2, Z, levels=50, cmap='viridis')
plt.colorbar(contour)
plt.plot(min_point[0], min_point[1], 'ro')  # Точка минимума
plt.title("Контурная карта функции")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid()
plt.show()
