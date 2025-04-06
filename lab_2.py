import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

# Параметры задачи (вариант 4)
gamma0 = -1        # u(0)
gamma1 = 1.38294  # u(1) (≈ (2)*ln2)
N = 100
h = 1 / N
x = np.linspace(0, 1, N+1)

# Правая часть
def f(x):
    return (x**2 + 2*x + 2) / (x + 1)

# Формирование СЛАУ для внутренних узлов (i=1,2,...,N-1)
# Размер матрицы будет (N-1) x (N-1)
A = np.zeros((N-1, N-1))
F = np.zeros(N-1)

for i in range(1, N):
    xi = x[i]
    # Коэффициенты для i-го уравнения (для u[i-1], u[i], u[i+1])
    a = 1/h**2 - (xi+1)/(2*h)    # коэффициент перед u_{i-1}
    b = -2/h**2 - 1              # коэффициент перед u_{i}
    c = 1/h**2 + (xi+1)/(2*h)     # коэффициент перед u_{i+1}
    idx = i - 1  # индекс в СЛАУ

    if i != 1:
        A[idx, idx-1] = a
    A[idx, idx] = b
    if i != N-1:
        A[idx, idx+1] = c
    # Правая часть
    F[idx] = f(xi)

# Корректировка F с учётом граничных условий:
# Для первого уравнения (i=1): влияние u[0]=gamma0
F[0] -= (1/h**2 - (x[1]+1)/(2*h)) * gamma0
# Для последнего уравнения (i=N-1): влияние u[N]=gamma1
F[-1] -= (1/h**2 + (x[N-1]+1)/(2*h)) * gamma1

# Решение системы
u_inner = np.linalg.solve(A, F)

# Полное решение: добавляем граничные условия
u = np.zeros(N+1)
u[0] = gamma0
u[1:N] = u_inner
u[N] = gamma1

# Точное решение (для сравнения)
u_exact = (x + 1) * np.log(x + 1)

# Вычисляем абсолютную погрешность
error = np.abs(u - u_exact)

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(x, u, label='Численное решение (центральные разности)')
plt.plot(x, u_exact, '--', label='Точное решение')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Решение краевой задачи (вариант 4)')
plt.legend()
plt.grid(True)
plt.show()

# Вывод таблицы результатов
print("\nТаблица результатов:")
print("    x      | Численное    | Точное       | Погрешность")
print("-----------+--------------+--------------+------------")
indices = np.linspace(0, N, 11, dtype=int)
for i in indices:
    print(f"{x[i]:.8f} | {u[i]:.8f} | {u_exact[i]:.8f} | {error[i]:.8e}")
