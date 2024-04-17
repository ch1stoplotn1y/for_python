import math
import sympy as sp
from sympy.abc import x,k
from sympy import sin, cos
import matplotlib.pyplot as plt
import numpy as np

l = 6 #float(input("Длина балки, м: "))
Iy = 135.88 / pow(10,8) #float(input("Момент инерции сечения, см^4:"))
A = 22  / pow(10,4)#float(input("Площадь поперечного сечения, см^2: "))
E = 206.01 * pow(10,9) #float(input("Модуль Юнга, ГПа: "))
M = 77008.5 #float(input("Объемный вес, H/м^3"))
print("Шарнир-шарнир")
N = 5
qp = 1.3 * 1000
qv = M * A
q = qv + qp
default_point = 1

fi = [sin((2*k-1) * math.pi * x / l) for k in range(N + 1)]
w = [sp.symbols("w" + str(k)) for k in range(N + 1)]
mult_w_fi = [w[k] * fi[k] for k in range(N + 1)]
new_w = [sp.simplify("0") for _ in range(N + 1)]
E_s = [0 for _ in range(N + 1)]
new_w_result = [[0 for _ in range(N + 1)] for _ in range(2 + 1)]

M_y = [sp.simplify("0") for _ in range(N + 1)]  # My(w, x)
M_y_of_x = [sp.simplify("0") for _ in range(N + 1)]  # My(x)
Q_y = [sp.simplify("0") for _ in range(N + 1)]  # Qy(w, x)
Q_y_of_x = [sp.simplify("0") for _ in range(N + 1)]  # Qy(x)

for k in range(1, N + 1):
    new_w[k] = sum(mult_w_fi[1:k + 1])
    E_s[k] = 1 / 2 * sp.integrate(E * Iy * sp.diff(new_w[k], x, 2) ** 2, (x, 0, l)) \
              - sp.integrate(q * new_w[k], (x, 0, l))
    grad_E_s = [sp.diff(E_s[k], w[_ + 1]) for _ in range(k)]
    grad_system_solution = sp.solve(grad_E_s, w[1:k + 1], dict = True) #решение системы градиент равно 0
    tuples_list_for_w = [tuple(
        [w[_ + 1], grad_system_solution[0][w[_ + 1]]]
    ) for _ in range(k)]    #составление списка кортежей wi для подстановки в выражение w__roof(wi, x)
    result_for_w = new_w[k].subs(tuples_list_for_w)     #получение new_w(x) из new_w(wi, x)

    M_y[k] = -E * Iy * sp.diff(new_w[k],x,2)
    M_y_of_x[k] = M_y[k].subs(tuples_list_for_w)
    Q_y[k] = -E * Iy * sp.diff(new_w[k],x,3)
    Q_y_of_x[k] = Q_y[k].subs(tuples_list_for_w)

    new_w_result[0][k] = result_for_w.subs(x, l / 2)  # подстановка wi и получение приближений для l/2
    new_w_result[1][k] = result_for_w.subs(x, l / 4)  # подстановка wi и получение приближений для l/4
    new_w_result[2][k] = result_for_w.subs(x, default_point)  # для пользовательской точки

w_of_x_result = q * x / (24 * E * Iy) * (x ** 3 - 2 * l * x ** 2 + l ** 3)  # точная формула прогиба балки

print("             Значение прогиба, м",)
print("Точка балки, м       Точное                Приближение(Ритц)")
print("                      w(x)  ",)
print("                                      new_w[1]       new_w[3]        new_w[5]")

print("   {:1.1f}               {:0.6f}      {:0.6f}     {:0.6f}      {:0.6f}   ".format(l / 2,
w_of_x_result.subs(x, l / 2),
new_w_result[0][1],
new_w_result[0][3],
new_w_result[0][5]))

print("   {:1.1f}               {:0.6f}       {:0.6f}      {:0.6f}       {:0.6f}   ".format(l / 4,
w_of_x_result.subs(x, l / 4),
new_w_result[1][1],
new_w_result[1][3],
new_w_result[1][5]))

print("   {:1.1f}               {:0.6f}       {:0.6f}      {:0.6f}       {:0.6f}   ".format(default_point,
w_of_x_result.subs(x,default_point),
new_w_result[2][1],
new_w_result[2][3],
new_w_result[2][5]))

x_w = np.arange(0,l,0.01)
y_w = [w_of_x_result.subs(x, x_w[i]) for i in range(len(x_w))]
plt.plot(x_w, y_w)
plt.ylabel("Прогиб w, м")
plt.title("Эпюра прогиба балки")
plt.show()

x_m = np.arange(0, l, 0.01)
y_m_2 = [M_y_of_x[1].subs(x, x_m[i]) for i in range(len(x_m))]
y_m_3 = [M_y_of_x[2].subs(x, x_m[i]) for i in range(len(x_m))]
y_m_4 = [M_y_of_x[3].subs(x, x_m[i]) for i in range(len(x_m))]
y_m_5 = [M_y_of_x[4].subs(x, x_m[i]) for i in range(len(x_m))]
y_m_6 = [M_y_of_x[5].subs(x, x_m[i]) for i in range(len(x_m))]
y_m_plot_list = [y_m_2,y_m_3,y_m_4,y_m_5,y_m_6]
for plot in y_m_plot_list:
    plt.plot(x_m, plot)
plt.title("Эпюра изгибающего момента")
plt.ylabel("Изгибающий момент My, Н * м")
plt.legend([str(i) for i in range(N)])
plt.subplots_adjust(hspace = 2)
plt.show()

x_l = np.arange(0,l,0.01)
y_l_2 = [Q_y_of_x[1].subs(x, x_l[i]) for i in range(len(x_l))]
y_l_3 = [Q_y_of_x[2].subs(x, x_l[i]) for i in range(len(x_l))]
y_l_4 = [Q_y_of_x[3].subs(x, x_l[i]) for i in range(len(x_l))]
y_l_5 = [Q_y_of_x[4].subs(x, x_l[i]) for i in range(len(x_l))]
y_l_6 = [Q_y_of_x[5].subs(x, x_l[i]) for i in range(len(x_l))]
y_l_plot_list = [y_l_2,y_l_3,y_l_4,y_l_5,y_l_6]
for plot in y_l_plot_list:
    plt.plot(x_l, plot)
plt.title("Перерезывающая сила")
plt.ylabel("Перерезывающая сила Qy, Н")
plt.legend([str(i) for i in range(N)])
plt.subplots_adjust(hspace = 2)
plt.grid()
plt.show()

iterations_plots = [
    plt.plot([_ for _ in range(1, N + 1)], [w_of_x_result.subs(x, l / 2) for i in range(1, N + 1)]),
    plt.plot([_ for _ in range(1, N + 1)], new_w_result[0][1:])]
iterations_plots[0].extend(iterations_plots[1])
plt.title("Сходимость итерационного процесса")
plt.ylabel("Прогиб w, м")
plt.xlabel("Номер приближения N")
plt.grid(visible=True)
plt.show()
