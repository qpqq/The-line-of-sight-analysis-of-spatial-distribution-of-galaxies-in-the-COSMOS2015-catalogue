def vs():
    import math
    from data_processing import z_final, z_min_chi2
    from graph import z_vs_z
    from random import sample
    from path import output_name

    def z_med_pdz(min_z=0, max_z=6, w=-100, err=False):
        from math import log10
        file = open(output_name, 'r')

        for i in range(2):
            file.readline()

        data = []
        for line in file:
            z = float(line[59:68]) * 10 ** int(line[69:73])
            z_f = float(line[44:53]) * 10 ** int(line[54:58])

            if min_z < z < max_z and min_z < z_f < max_z and float(line[119:128]) * 10 ** int(line[129:132]) > w:
                if err:
                    z_min = z - float(line[74:83]) * 10 ** int(line[84:88])
                    z_max = float(line[89:98]) * 10 ** int(line[99:103]) - z
                    data.append([log10(z), z_min / z, z_max / z])
                else:
                    data.append(z)

        file.close()
        return sorted(data)

    def kk(data, delta, k):
        n = int((data[-1][0] - data[0][0]) / delta)
        if n - (data[-1][0] - data[0][0]) / delta != 0:
            n += 1
        bins = [data[0][0] + i * delta for i in range(n + 1)]

        j = 0
        new_data = [[], [[], []]]
        for i in range(1, len(bins)):
            d = []
            while j < len(data) and data[j][0] < bins[i]:
                d.append(data[j])
                j += 1
            sum_z, sum_err_max, sum_err_min = 0, 0, 0
            for t in d:
                sum_z += t[0]
                sum_err_min += t[1]
                sum_err_max += t[2]
            if len(d) == 0:
                continue
            new_data[0].append(sum_z / len(d))
            new_data[1][0].append(sum_err_min / len(d))
            new_data[1][1].append(sum_err_max / len(d))
            '''
            for t in dd:
                new_data[0].append(t[0])
                new_data[1][0].append(t[1])
                new_data[1][1].append(t[2])
            '''

        return new_data

    w = 0.7
    # z_chi = [[k, 0, 0] for k in z_min_chi2(w=w)]
    z_fin = z_final(w=w, err=True)
    z_med = z_med_pdz(w=w, err=True)

    dz = math.log10(6) / 100
    p = 10
    # z_chi = kk(z_chi, dz, p)
    z_fin = kk(z_fin, dz, p)
    z_med = kk(z_med, dz, p)

    z_vs_z(z_fin[0], z_fin[1], z_med[0], z_med[1])


'''
def app_graph(func, data, n, alpha, w):
    import approx
    import numpy as np
    from math import exp

    print('n={}, w>{}'.format(str(n), str(w)))

    x, y = data[0], data[1]
    a1, b1, c1, s1 = approx.least_squares_1_10(x, y)
    a2, b2, z2, c2, s2 = approx.least_squares_2(x, y, 1)
    a3, b3, d3, c3, s3 = approx.least_squares_3(x, y, 1)

    x0 = np.linspace(x[0], x[-1], 500)

    def y1(t):
        return c1 * t ** a1 * exp(-b1 * t)

    def y2(t):
        return c2 * t ** a2 * exp(-t / z2) ** b2

    def y3(t):
        return c3 * (t ** a3 + t ** (a3 * b3)) / (t ** b3 + d3)

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    # ax.set_xscale('log')
    # ax.set_yscale('log')

    ax.scatter(x, [k / sum(y) for k in y], s=10, c=[[1, 0, 1] for i in x])
    ax.plot(x, [k / sum(y) for k in y])

    label1 = '$' + str(int(c1)) + 'x^{' + str(a1) + '}e^{-' + str(b1) + 'x}$'
    label1 += ', ' + '%.3e' % s1
    ax.plot(x0, [y1(i) for i in x0], color='orange', linestyle='--', linewidth=0.8, label=label1)

    label2 = '$' + str(int(c2)) + 'x^{' + str(a2) + r'}e^{\left(-\frac{x}{' + str(z2) + r'}\right)^{' + str(b2) + '}}$'
    label2 += ', ' + '%.3e' % s2
    ax.plot(x0, [y2(i) / sum(y) for i in x0], color='limegreen', linestyle='--', linewidth=0.8, label=label2)

    label3 = '$' + str(int(c3)) + r'\left(\frac{x^{' + str(a3) + '}+x^{' + str(a3) + r'\cdot' + str(b3) + '}}{x^{' \
             + str(b3) + '}+' + str(d3) + r'}\right)$'
    label3 += ', ' + '%.3e' % s3
    ax.plot(x0, [y3(i) / sum(y) for i in x0], color='orangered', linestyle='--', linewidth=0.8, label=label3)

    ax.set_xlabel('z')
    ax.set_ylabel('N(z)')

    ax.set_xlim(0)
    ax.set_ylim(0)

    if w == -100:
        plt.title('Approximations for {} with n={}, alpha={}, all w, N={:d}'.format(func.upper(), str(n), str(alpha), sum(y)))
    else:
        plt.title('Approximations for {} with n={}, alpha={}, w>{}, N={:d}'.format(func.upper(), str(n), str(alpha), str(w), sum(y)))
    ax.legend(fontsize='x-large')
    filename = 'n={:d} alpha={}.png'.format(n, str(alpha))
    # plt.savefig(filename, dpi=150)
    plt.show()
    plt.close()


w = 0.7
n = 100
alpha = 2
data = data_log(z_final(w=w), n, alpha, 'center')
app_graph('z_final', data, n, alpha, w)
'''