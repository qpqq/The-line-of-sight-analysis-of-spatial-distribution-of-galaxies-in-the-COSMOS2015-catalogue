from scipy import integrate
from scipy.optimize import leastsq
from math import exp, log10
from random import uniform, randint, random
import numpy as np


def least_squares_1(x, y, delta):
    s = 10 ** 100
    delta = 10 ** delta

    sum_y = sum(y) * (x[-1] - x[0]) / (len(x) - 1)
    ans = []

    def f(t):
        return t ** a * exp(-b * t)

    for a in range(4 * delta + 1):
        a /= delta

        for b in range(4 * delta + 1):
            b /= delta

            diff = []
            const = sum_y / integrate.quad(f, 0, 6)[0]

            for i in range(len(x)):
                diff.append((y[i] - const * f(x[i])) ** 2)

            sum_diff = sum(diff)

            if s > sum_diff:
                s = sum_diff
                ans = [a, b, const, s]

    return ans


def least_squares_2(x, y, delta):
    s = 10 ** 100
    delta = 10 ** delta

    sum_y = sum(y) * (x[-1] - x[0]) / (len(x) - 1)
    ans = []

    def f(t):
        return t ** a * exp(-t / z) ** b

    for a in range(3 * delta + 1):
        a /= delta
        print(a, end=' ')

        for b in range(5, 7 * delta + 1):
            b /= delta

            for z in range(1, 4 * delta + 1):
                z /= delta

                diff = []
                const = sum_y / integrate.quad(f, 0, 6)[0]

                for i in range(len(x)):
                    diff.append((y[i] - const * f(x[i])) ** 2)

                sum_d = sum(diff)

                if s > sum_d:
                    s = sum_d
                    ans = [a, b, z, const, s]

    print()
    return ans


def least_squares_3(z3, y3, delta):
    s = 10 ** 100
    delta = 10 ** delta

    sum_y = sum(y3) * (z3[-1] - z3[0]) / (len(z3) - 1)
    ans = []

    def f(t):
        return (t ** a + t ** (a * b)) / (t ** b + c)

    for a in range(1, 3 * delta + 1):
        a /= delta
        print(a, end=' ')

        for b in range(1, 6 * delta + 1):
            b /= delta

            for c in range(1, 50):
                c /= delta

                diff = []
                const = sum_y / integrate.quad(f, 0, 6)[0]

                for i in range(len(z3)):
                    diff.append((y3[i] - const * f(z3[i])) ** 2)

                su = sum(diff)

                if s > su:
                    s = su
                    ans = [a, b, c, const, s]

    print()
    return ans


def var():
    for i in range(1, 100):
        yield i / 100
    for i in range(10, 101):
        yield i / 10


def least_squares_1_10(x, y):
    s = 10 ** 100

    sum_y = sum(y) * (x[-1] - x[0]) / (len(x) - 1)
    ans = []

    def f(t):
        return t ** a * exp(-b * t)

    for a in var():
        print(a, end=' ')
        for b in var():
            diff = []
            const = sum_y / integrate.quad(f, 0, 6)[0]

            for i in range(len(x)):
                diff.append((y[i] / sum(y) - const * f(x[i])) ** 2)

            sum_diff = sum(diff)
            if s > sum_diff:
                s = sum_diff
                ans = [a, b, const, s]

    print()
    return ans


def least_squares_2_10(x, y):
    s = 10 ** 100

    sum_y = sum(y) * (x[-1] - x[0]) / (len(x) - 1)
    ans = []

    def f(t):
        return t ** a * exp(-t / z) ** b

    for a in var():
        print(a, end=' ')
        for b in var():
            for z in var():
                diff = []
                const = sum_y / integrate.quad(f, 0, 6)[0]

                for i in range(len(x)):
                    diff.append((y[i] - const * f(x[i])) ** 2)

                sum_d = sum(diff)
                if s > sum_d:
                    s = sum_d
                    ans = [a, b, z, const, s]

    print()
    return ans


def f(t, par, kind):
    t = np.array(t)

    if kind == 1:
        alpha, beta, const = par
        return const * t ** alpha * np.exp(-beta * t)
    elif kind == 2:
        alpha, beta, zeta, const = par
        return const * t ** alpha * np.exp(- t / zeta) ** beta
    elif kind == 3:
        alpha, beta, delta, const = par
        if beta > 390 or alpha * beta > 390:
            return [10 ** 19] * len(t)
        return const * (t ** alpha + t ** (alpha * beta)) / (t ** beta + delta)


def y_app(bins, dz, par, kind):
    yapp = []
    for i in range(1, len(bins)):
        yapp.append(integrate.quad(f, bins[i - 1], bins[i], args=(par, kind))[0] / dz)
    return np.array(yapp)


def fast(x, y, kind):
    # from math import sqrt

    x, y, = np.array(x), np.array(y)
    # xb = np.array([0] + [k + x[0] for k in x])

    def diff(parameters):
        # y_f = y_app(xb, x[0] * 2, parameters, kind)
        # y_fluct = (y - y_f) / y_f
        # s_plus = 0
        # s_minus = 0
        # for k in y_fluct:
        #     if k >= 0:
        #         s_plus += k ** 2
        #     else:
        #         s_minus += k ** 2
        # if s_plus == 0 or s_minus == 0:
        #     return (y - f(x, parameters, kind)) + 10 ** 15
        # else:
        #     return np.log(abs(y - f(x, parameters, kind))) * np.sqrt(abs(1 - np.log(s_minus) / np.log(s_plus)))

        return y - f(x, parameters, kind)

    if kind == 1:
        p = np.array([1, 2, 10 ** 5])

    elif kind == 2:
        p = np.array([1, 1, 1, 10 ** 4])

    elif kind == 3:
        p = np.array([0.8, 5, 0.7, 10000])
        '''
        много чего: [0.9, 2, 0.9, 50000]
        w>0.9: [0.8, 8, 0.9, 40000]
        w>0.97: [0.8, 8, 0.9, 40000]
        dz=0.3, w>0.8: [0.7, 7, 0.99, 50000] нет(?)
        '''

    else:
        print('Kind error')
        return

    par = leastsq(diff, p, maxfev=10000)[0]

    # y_fluct_2 = (y - y_app(xb, x[0] * 2, par, kind)) / y_app(xb, x[0] * 2, par, kind)
    # s_plus_2 = 0
    # s_minus_2 = 0
    # for k in y_fluct_2:
    #     if k >= 0:
    #         s_plus_2 += k ** 2
    #     else:
    #         s_minus_2 += k ** 2
    # if s_plus_2 == 0:
    #     return par, 0
    # else:
    #     return par, abs(1 - s_minus_2 / s_plus_2)

    err = (y - f(x, par, kind)) ** 2
    return par, sum(err)


def test_1(n):
    def f(v):
        return v ** a1 * exp(-b1 * v)

    a1 = uniform(0, 10)
    b1 = uniform(0, 10)
    const1 = uniform(10 ** 5, 10 ** 6)

    x = np.linspace(0, 6, n)
    y = [const1 * f(u) for u in x]

    print(a1, b1, const1)
