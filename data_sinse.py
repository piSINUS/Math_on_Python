from sympy import *
from sympy.plotting import plot
from sympy.solvers.inequalities import solve_univariate_inequality
from sympy.plotting import plot3d
init_printing(use_unicode=False, wrap_line = False, no_global= True)
import numpy as np
import matplotlib.pyplot as plt
# # для построения граaиков
# x = Symbol('x')
# f  = x**2
# f2 = 2*x**2 + 10*x -12
# f3 = sqrt(x)
# f4 = ln(x)
# plot(f3, (x,0,100))
##для решения уравнений
# x= Symbol('x')
# f=  x**2 +3*x - 4 нули квадр aункции
# print(solve(f),'- nulls of func')
##для решения неравенств
# x = Symbol('x')
# f = x**2 + 3 * x -4
# print(solve_univariate_inequality(f>0,x))
# для решения систем уравнений  (линейных и не линейных)
# a5, a4, a3, a2, a1, a0 =symbols(' a5, a4, a3, a2, a1, a0')

# eq1_lp =(-3)**5*a5+(-3)**4*a4+(-3)**3*a3+(-3)**2*a2+(-3)*a1+a0-33
# eq2_lp =(-2)**5*a5+(-2)**4*a4+(-2)**3*a3+(-2)**2*a2+(-2)*a1+a0-31
# eq3_lp =(-1)**5*a5+(-1)**4*a4+(-1)**3*a3+(-1)**2*a2+(-1)*a1+a0-18
# eq4_lp =(1)**5*a5+(1)**4*a4+(1)**3*a3+(1)**2*a2+(1)*a1+a0+18
# eq5_lp =(2)**5*a5+(2)**4*a4+(2)**3*a3+(2)**2*a2+(2)*a1+a0+31
# eq6_lp =(3)**5*a5+(3)**4*a4+(3)**3*a3+(3)**2*a2+(3)*a1+a0+33

# a = nonlinsolve([eq1_lp, eq2_lp, eq3_lp,eq4_lp, eq5_lp, eq6_lp],[a5, a4, a3, a2, a1, a0])

# print(a)

# print(round((1.258**2) + (0.24**2) + (1.151**2))/3)
# print(((3.688-3.75)**2 + (10.791-8.25)**2 + (20.705-23.25)**2)/3)

# print(0.5*x**3-0.25*x**2+0.75*x+1.25)
# print(((4.872-5.75)**2+(29.707-32.25)**2+(485.724-483.75)**2+(246.971-247.25)**2+(840.658-838.25)**2)/5)


# Для сдвигов aункции 
# def print_points_ands_function(sympy_function):
#     def function(x_): return float(sympy_function.subs(x,x_))

#     points_X = np.array([-3,-2,-1,1,2,3,])
#     points_Y = np.array([15.0, 8.0, 13.0, 10.0, 15.0, 30.0])
#     plt.xlim(-25, 7.5)
#     plt.ylim(-1,40)

#     plt.scatter(points_X, points_Y,c = 'r')
#     x_range = np.linspace(plt.xlim()[0], plt.xlim()[1], num = 100)
#     function_Y = [function(x_) for x_ in x_range]
#     plt.p200lot200(x_range, function_Y, 'b')
#     plt.show

#     MSE = sum([(points_Y[i] - function(points_X[i]))**2 for i in range(len(points_Y))]) / len(points_Y)
#     print(f'MSE = {MSE}')

# x = Symbol('x')
# f = x**2 + 32*x +265


# f_new =  f.subs(x, x-15)
# f_new2 = f_new.subs(x, x*1.12)
# print_points_ands_function(f_new2)
# plot(f_new2)
# Производная  
# x = Symbol('x')
# diff_f = 3 * x **2 - 4 * x +1
# print(solve_univariate_inequality(diff_f < 0, x))
# print(solve(diff_f))
# 
# Для граaиков с несколькими переменными
# x, y  = symbols('x, y')
# # f =  x*y / sqrt(x**2 +  y**2) - 5
# # plot3d(f)
# f =  x*y / sqrt(x**2 +  y**2-5) 
# print(solve(f))
# plot3d(f )
# a2, a1, a0 = symbols('a2, a1, a0')

# mse = 1/3*(((a2 * 2 + a1 *200 + a0) - 200)**2 + \
#     ((a2 * 1 + a1 * 450 + a0)-300)**2 + \
#     ((a2 *3 + a1 * 550 +a0) - 600)**2)
# msea2 = 1/3*(28*a2+5000*a1+12*a0-5000)
# msea1 = 1/3*(5000*a2+1090000*a1+2400*a0-1010000)
# msea0 = 1/3*(12*a2+2400*a1+6*a0-2200)

# print(nonlinsolve([msea2,msea1,msea0],[a2,a1,a0]))
# print(mse.subs({a2:108.333333333333,a1:0.833333333333351,a0:-183.33333333334}))
# x2,x1 = symbols('x2 , x1')
# f = 325/3*x2+2.5/3*x1-550/3
# print(f.subs({x1:500,x2:4}))

#считаем производные

# x = Symbol('x')
# th = (exp(x) -  exp(-x)) / (exp(x)+exp(-x))
# print(diff(mse,a1))
# print(diff(th))
# 
# Полный порядок нахождения MSE для чисел Масса(a1x1)1945,1495,1570,1520; Мощность двигателя(a2,x2) 560,340,343,431; Время разгона(y) 4,.4, 4.9, 5.2 ,?:
# MSE = (sum(Истенное значение - наше предсказание)**2)колво эл
a2, a1, a0 = symbols('a2, a1, a0')#1 задаём зависящие переменные
x2,x1 = symbols('x2 , x1')
mse = 1/3*(((a2 * 560 + a1 *1945 + a0) - 4.3)**2 + \
    ((a2 * 340 + a1 * 1495 + a0) - 4.9)**2 + \
    ((a2 *343 + a1 * 1570 +a0) - 5.2)**2)#2 Определяем MSE 

mseDifA2 = diff(mse,a2)#Находим частные производные с помощью diff
mseDifA1 = diff(mse,a1)
mseDifA0 = diff(mse,a0)
#Теперь ищем точку минимума, необходимое условие экстремума: частные производные должны быть равны 0. Решаем систему уравнений с помощью nonlinsolve
print(nonlinsolve([mseDifA2,mseDifA1,mseDifA0], [a2,a1,a0]))#получаем нули функции
# мы нашли значения параметров теперь подставляем их в MSE
print(mse.subs({'a2':-0.0118811881188827, 'a1': 0.00447524752478956, 'a0' : 2.24910891085849}))
#МЫ ПОЛУЧИЛИ ОЧЕНЬ МАЛЕНЬКОЕ мсе 1.10602959755638e-24, но для проверки поменяем знак у a1
print(mse.subs({'a2':-0.0118811881188827, 'a1': -0.00447524752478956, 'a0' : 2.24910891085849}))
#Мы получили гораздл большее MSE => мы нашли минима льную сейчас построим её 
funckMse = -0.0118811881188827 * x2 + 0.00447524752478956*x1 + 2.24910891085849
# теперь можем предсказать время разгона авто с мощностью двигателя 431 лс и массой 1520 .Подставляем с помощью subs
print(funckMse.subs({'x2': 431, 'x1': 1520}))
# получаем время разгона 3.93069306930018