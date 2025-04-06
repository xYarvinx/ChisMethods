import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from scipy.interpolate import RegularGridInterpolator

def solve_wave(N, M, L=1.0, T=1.0, g0=lambda t: -2*t, g1=lambda t: -t**2,
               y=lambda x: 0.0, v=lambda x: 0.0):
    """
    Решает волновое уравнение u_tt = u_xx на прямоугольнике [0,L]x[0,T]
    с граничными условиями u(0,t)=g0(t), u(L,t)=g1(t) и начальными условиями
    u(x,0)=y(x), u_t(x,0)=v(x) с использованием центральной разностной схемы.
    """
    h = L / N
    tau = T / M
    x = np.linspace(0, L, N+1)
    t = np.linspace(0, T, M+1)

    U = np.zeros((M+1, N+1))
    # Начальное условие
    for i in range(N+1):
        U[0, i] = y(x[i])

    # Граничные условия для всех времых уровней
    for n in range(M+1):
        U[n, 0] = g0(t[n])
        U[n, -1] = g1(t[n])

    # Первый временной слой (n=0 -> n=1)
    for i in range(1, N):
        U[1, i] = U[0, i] + tau*v(x[i]) + 0.5*(tau**2)/h**2 * (U[0, i+1] - 2*U[0, i] + U[0, i-1])

    r = (tau/h)**2
    for n in range(1, M):
        for i in range(1, N):
            U[n+1, i] = 2*U[n, i] - U[n-1, i] + r*(U[n, i+1] - 2*U[n, i] + U[n, i-1])

    return x, t, U

# Параметры для варианта 4
g0 = lambda t: -2*t
g1 = lambda t: -t**2
y_init = lambda x: 0.0
v_init = lambda x: 0.0

# Решение на грубой сетке (N=M=10)
N_coarse = 10
M_coarse = 10
x_coarse, t_coarse, U_coarse = solve_wave(N_coarse, M_coarse, g0=g0, g1=g1, y=y_init, v=v_init)

# Решение на тонкой (эталонной) сетке (N=M=100)
N_fine = 100
M_fine = 100
x_fine, t_fine, U_fine = solve_wave(N_fine, M_fine, g0=g0, g1=g1, y=y_init, v=v_init)

# Интерполируем эталонное решение на узлы грубой сетки
# Создаем интерполятор с регулярной сеткой (ось t, затем x)
interp_func = RegularGridInterpolator((t_fine, x_fine), U_fine)

U_fine_on_coarse = np.zeros_like(U_coarse)
for n in range(M_coarse+1):
    # Для фиксированного t_coarse[n] интерполируем по всем x_coarse
    pts = np.column_stack((np.full(x_coarse.shape, t_coarse[n]), x_coarse))
    U_fine_on_coarse[n, :] = interp_func(pts)

# Вычисляем погрешность на финальном временном слое (t = T)
error = np.abs(U_coarse[-1, :] - U_fine_on_coarse[-1, :])
max_error = np.max(error)
print("Погрешность (max-norm) на финальном слое t = 1:", max_error)

# График решения на финальном временном слое
plt.figure(figsize=(8,5))
plt.plot(x_coarse, U_coarse[-1, :], 'o-', label='Грубая сетка (N=M=10)')
plt.plot(x_coarse, U_fine_on_coarse[-1, :], 's--', label='Интерполированный эталон (N=M=100)')
plt.xlabel("x")
plt.ylabel("u(x, T=1)")
plt.title("Решение задачи колебаний струны, вариант 4")
plt.legend()
plt.grid(True)
plt.show()

# Таблица результатов
print("\n   x    |  Грубая  |  Эталон (интерполирован) |  Погрешность")
print("---------------------------------------------------------------")
for i in range(N_coarse+1):
    print(f"{x_coarse[i]:6.3f}  | {U_coarse[-1, i]:9.5f}  | {U_fine_on_coarse[-1, i]:9.5f}  | {error[i]:9.5e}")
