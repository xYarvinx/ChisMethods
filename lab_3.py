import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

# ----------------------------
# Параметры задачи
# ----------------------------
L = 1.0          # длина отрезка по x (x ∈ [0,1])
N = 20           # число разбиений по x, таким образом, число узлов = N+1
h = L / N        # шаг по x
tau = 0.001      # шаг по времени (с учетом условия устойчивости для явной схемы)
M = 6            # число временных шагов
T = M * tau      # конечное время

# Сетка по x
x = np.linspace(0, L, N+1)

# ----------------------------
# Функции задачи (вариант 4)
# ----------------------------
def j(x, t):
    # j(x,t) = x*(x - x^2) = x^2 - x^3 (нет явной зависимости от t)
    return x**2 - x**3

def y(x):
    # начальное условие: u(x,0)= x - x^2
    return x - x**2

def g0(t):
    return 0.0

def g1(t):
    return 0.0

# ----------------------------
# Метод 1. Явная схема (forward in time)
# ----------------------------
def explicit_method():
    # Инициализируем решение: u^0_i = y(x_i)
    U = np.zeros((M+1, N+1))
    U[0, :] = y(x)
    # Граничные условия для всех временных слоев
    U[:, 0] = g0(0)  # здесь g0(t)=0
    U[:, -1] = g1(0) # g1(t)=0

    # Явная схема: для i=1..N-1, n=0..M-1
    for n in range(0, M):
        t_n = n * tau
        for i in range(1, N):
            # центральная разность для u_xx
            u_xx = (U[n, i+1] - 2*U[n, i] + U[n, i-1]) / h**2
            U[n+1, i] = U[n, i] + tau * (u_xx + j(x[i], t_n))
        # Обновляем граничные условия
        U[n+1, 0] = g0(t_n + tau)
        U[n+1, -1] = g1(t_n + tau)
    return U

# ----------------------------
# Метод 2. Чисто неявная схема (backward Euler)
# ----------------------------
def implicit_method():
    U = np.zeros((M+1, N+1))
    U[0, :] = y(x)
    U[:, 0] = g0(0)
    U[:, -1] = g1(0)

    # Коэффициенты для СЛАУ для внутренних узлов:
    # - (tau/h^2)*u_{i-1}^{n+1} + (1 + 2*tau/h^2)*u_i^{n+1} - (tau/h^2)*u_{i+1}^{n+1} = u_i^n + tau*j(x_i, t_{n+1})
    A = np.zeros((N-1, N-1))
    alpha = -tau / h**2
    beta  = 1 + 2*tau / h**2
    for i in range(N-1):
        A[i, i] = beta
        if i > 0:
            A[i, i-1] = alpha
        if i < N-2:
            A[i, i+1] = alpha

    for n in range(0, M):
        t_next = (n+1)*tau
        F = np.zeros(N-1)
        for i in range(1, N):
            F[i-1] = U[n, i] + tau * j(x[i], t_next)
        # Граничные условия влияют только если g0 или g1 не равны нулю (тут они равны 0)
        U_inner = np.linalg.solve(A, F)
        U[n+1, 1:N] = U_inner
        U[n+1, 0] = g0(t_next)
        U[n+1, -1] = g1(t_next)
    return U

# ----------------------------
# Метод 3. Схема Кранка–Николсона
# ----------------------------
def crank_nicolson():
    U = np.zeros((M+1, N+1))
    U[0, :] = y(x)
    U[:, 0] = g0(0)
    U[:, -1] = g1(0)

    # Параметр для усреднения по пространственной разности
    gamma = tau / (2 * h**2)
    # Формируем матрицу A для левого (неизвестного) члена:
    # u_i^{n+1} - gamma*(u_{i+1}^{n+1} - 2*u_i^{n+1} + u_{i-1}^{n+1})
    # => коэффициенты: diag: 1 + tau/h^2, соседние: -tau/(2h^2)
    A = np.zeros((N-1, N-1))
    for i in range(N-1):
        A[i, i] = 1 + tau/h**2
        if i > 0:
            A[i, i-1] = -gamma
        if i < N-2:
            A[i, i+1] = -gamma

    for n in range(0, M):
        t_n = n * tau
        t_np1 = (n+1) * tau
        # Формируем правую часть:
        # F_i = u_i^n + gamma*(u_{i+1}^n - 2*u_i^n + u_{i-1}^n) + tau * j(x_i, (t_n+t_np1)/2)
        F_vec = np.zeros(N-1)
        for i in range(1, N):
            F_vec[i-1] = U[n, i] + gamma * (U[n, i+1] - 2*U[n, i] + U[n, i-1]) \
                         + tau * j(x[i], (t_n+t_np1)/2)
        U_inner = np.linalg.solve(A, F_vec)
        U[n+1, 1:N] = U_inner
        U[n+1, 0] = g0(t_np1)
        U[n+1, -1] = g1(t_np1)
    return U

# ----------------------------
# Решение задач методами
# ----------------------------
U_explicit = explicit_method()
U_implicit = implicit_method()
U_CN = crank_nicolson()

# ----------------------------
# Вычисление погрешностей
# ----------------------------
# Считаем, что решение, полученное по схеме Кранка–Николсона, является эталонным
error_explicit = np.max(np.abs(U_explicit[-1, :] - U_CN[-1, :]))
error_implicit = np.max(np.abs(U_implicit[-1, :] - U_CN[-1, :]))
print("Максимальная погрешность:")
print(f"Явная схема vs Кранк-Николсон: {error_explicit:.6e}")
print(f"Неявная схема vs Кранк-Николсон: {error_implicit:.6e}")

# ----------------------------
# Построение графиков финального временного слоя
# ----------------------------
plt.figure(figsize=(10,6))
plt.plot(x, U_explicit[-1, :], 'o-', label='Явная схема')
plt.plot(x, U_implicit[-1, :], 's-', label='Неявная схема')
plt.plot(x, U_CN[-1, :], '^-', label='Кранк–Николсон (эталон)')
plt.xlabel("x")
plt.ylabel("u(x, T)")
plt.title(f"Решение задачи для T = {T:.5f} (6 временных шагов)")
plt.legend()
plt.grid(True)
plt.show()

# Вывод значений решения на финальном слое (T = 6*tau)
print("\n     x      |   Явная    |   Неявная  |  Кранк–Николсон")
print("------------------------------------------------------------")
for i in range(N+1):
    print(f"{x[i]:6.3f}  | {U_explicit[-1,i]:10.6f}  | {U_implicit[-1,i]:10.6f}  | {U_CN[-1,i]:10.6f}")
