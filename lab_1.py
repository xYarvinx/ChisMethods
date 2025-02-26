import math


# Определение правой части дифференциального уравнения
def f(x, y, a, c):
    return a / (c - x)


# Точное решение
def exact_solution(x, a, c):
    return a * math.log(c / (c - x))


# Усовершенствованный метод ломаных
def improved_euler_method(f, x0, y0, xn, h, a, c):
    n = int((xn - x0) / h)  # Количество шагов
    x_values = [x0]
    y_values = [y0]

    for i in range(n):
        x_current = x_values[-1]
        y_current = y_values[-1]

        # Вычисляем k1 и k2
        k1 = f(x_current, y_current, a, c)
        k2 = f(x_current + h / 2, y_current + h / 2 * k1, a, c)

        # Обновляем значение y
        y_next = y_current + h * k2
        x_next = x_current + h

        # Добавляем значения в списки
        x_values.append(x_next)
        y_values.append(y_next)

    return x_values, y_values


# Параметры задачи
x0 = 0
y0 = 0
xn = 1
a = 1
c = 2

# Шаг h
h = 0.2

# Решение с шагом h
x_h, y_h = improved_euler_method(f, x0, y0, xn, h, a, c)
exact_h = [exact_solution(x, a, c) for x in x_h]
error_h = [abs(exact - approx) for exact, approx in zip(exact_h, y_h)]

# Решение с шагом h/2
h_half = h / 2
x_h_half, y_h_half = improved_euler_method(f, x0, y0, xn, h_half, a, c)
exact_h_half = [exact_solution(x, a, c) for x in x_h_half]
error_h_half = [abs(exact - approx) for exact, approx in zip(exact_h_half, y_h_half)]

# Вывод таблицы для шага h
print("Таблица для шага h = {:.2f}".format(h))
print("{:<10} {:<20} {:<20} {:<15}".format("x", "Точное решение", "Приближенное решение", "Погрешность"))
for i in range(len(x_h)):
    print("{:<10.2f} {:<20.6f} {:<20.6f} {:<15.6e}".format(x_h[i], exact_h[i], y_h[i], error_h[i]))

print("\nТаблица для шага h = {:.3f}".format(h_half))
print("{:<10} {:<20} {:<20} {:<15}".format("x", "Точное решение", "Приближенное решение", "Погрешность"))
for i in range(len(x_h_half)):
    print("{:<10.3f} {:<20.6f} {:<20.6f} {:<15.6e}".format(x_h_half[i], exact_h_half[i], y_h_half[i], error_h_half[i]))