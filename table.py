import pandas as pd
import numpy as np
import approx as ap
import distance as di
from math import log10
from path import input_name
from data_processing import data_hist


def bigdata():
    file = open(input_name, 'r')

    names = file.readline().split()[1:]
    file.readline()

    q0, q1, q2, q3, q4, q5, q6, q7, q8 = [], [], [], [], [], [], [], [], []
    index = []
    for line in file:
        index.append(int(line[7:13]))
        q0.append(float(line[14:23]) * 10 ** int(line[24:28]))
        q1.append(float(line[29:38]) * 10 ** int(line[39:43]))
        q2.append(float(line[44:53]) * 10 ** int(line[54:58]))
        q3.append(float(line[59:68]) * 10 ** int(line[69:73]))
        q4.append(float(line[74:83]) * 10 ** int(line[84:88]))
        q5.append(float(line[89:98]) * 10 ** int(line[99:103]))
        q6.append(float(line[105:113]) * 10 ** int(line[114:118]))
        q7.append(float(line[119:128]) * 10 ** int(line[129:132]))
        q8.append(float(line[133:142]) * 10 ** int(line[143:146]))

    data = {names[0]: q0,
            names[1]: q1,
            names[2]: q2,
            names[3]: q3,
            names[4]: q4,
            names[5]: q5,
            names[6]: q6,
            names[7]: q7,
            names[8]: q8
            }

    d = pd.DataFrame(data, index=index)
    d.index.name = 'ID'
    d.to_csv('pdz_cosmos2015_v1.3_t1.csv')
    b = pd.read_csv('pdz_cosmos2015_v1.3_t1.csv', index_col=0)
    print(b)


def y_app(bins, dz, par, kind):
    from scipy import integrate

    yapp = []
    for i in range(1, len(bins)):
        yapp.append(integrate.quad(ap.f, bins[i - 1], bins[i], args=(par, kind))[0] / dz)

    return np.array(yapp)


def my_main_old(z, delta, sigma):
    def start():
        number.append(j)
        count.append(1)
        z_start.append(z[i])
        start_err_com.append(0)
        start_err_pro.append(0)
        z_finish.append(z[i])
        finish_err_com.append(0)
        finish_err_pro.append(0)
        z_mid.append(0)
        r_com.append(0)
        r_pro.append(0)
        sigma_fluct.append(delta[i] ** 2)
        sigma_fluct_err.append(0)
        sigma_poisson.append(sigma[i])
        sigma_obs.append(0)

    def cont():
        count[j] += 1
        sigma_fluct[j] += delta[i] ** 2
        sigma_poisson[j] += sigma[i]

    def fin():
        z_finish[j] = z[i]
        start_err_com[j] = di.r_comoving(z_start[j] + dz) - di.r_comoving(z_start[j] - dz)
        start_err_pro[j] = di.r_proper(z_start[j] + dz) - di.r_proper(z_start[j] - dz)
        finish_err_com[j] = di.r_comoving(z_finish[j] + dz) - di.r_comoving(z_finish[j] - dz)
        finish_err_pro[j] = di.r_proper(z_finish[j] + dz) - di.r_proper(z_finish[j] - dz)
        z_mid[j] = (z_finish[j] + z_start[j]) / 2
        r_com[j] = di.r_comoving(z_finish[j]) - di.r_comoving(z_start[j])
        r_pro[j] = di.r_proper(z_finish[j]) - di.r_proper(z_start[j])
        sigma_fluct[j] /= count[j] - 1
        sigma_poisson[j] /= count[j] - 1
        sigma_obs[j] = sigma_fluct[j] - sigma_poisson[j]
        sigma_fluct[j] = np.sqrt(sigma_fluct[j])
        sigma_poisson[j] = np.sqrt(sigma_poisson[j])
        sigma_obs[j] = np.sqrt(sigma_obs[j])

        i_start = list(z).index(z_start[j])
        i_finish = list(z).index(z_finish[j])
        for k in range(i_start, i_finish + 1):
            sigma_fluct_err[j] += (delta[k] - sigma_fluct[j]) ** 2
        sigma_fluct_err[j] /= count[j] - 1
        sigma_fluct_err[j] = np.sqrt(sigma_fluct_err[j])

    sgn = []
    i = 0

    if np.sign(delta[i]) == np.sign(delta[i] + delta[i + 1]):
        sgn.append(np.sign(delta[i]))
    else:
        sgn.append(0.0)
    i += 1

    while i < len(delta) - 1:
        if np.sign(delta[i]) == np.sign(delta[i] + delta[i + 1]) == np.sign(delta[i - 1] + delta[i]):
            sgn.append(np.sign(delta[i]))
        else:
            sgn.append(0.0)
        i += 1

    if np.sign(delta[i]) == np.sign(delta[i - 1] + delta[i]):
        sgn.append(np.sign(delta[i]))
    else:
        sgn.append(0.0)

    dz = (z[1] - z[0]) / 2
    number = []
    count = []
    z_start = []
    start_err_com = []
    start_err_pro = []
    z_finish = []
    finish_err_com = []
    finish_err_pro = []
    z_mid = []
    r_com = []
    r_pro = []
    sigma_fluct = []
    sigma_fluct_err = []
    sigma_poisson = []
    sigma_obs = []

    i, j = 0, -1
    new = False
    if sgn[0] == 0:
        new = True
    else:
        j += 1
        start()

    for i in range(1, len(sgn) - 1):
        if new:
            if sgn[i + 1] == 0:
                continue

            new = False

            j += 1
            start()

        elif sgn[i] == 0:
            cont()
            fin()

            if sgn[i + 1] == 0:
                new = True
                continue

            j += 1
            start()

        else:
            cont()

    i += 1
    if not new:
        cont()
        fin()

    return number, count, z_start, start_err_com, start_err_pro, z_finish, finish_err_com, finish_err_pro, z_mid, \
        r_com, r_pro, sigma_fluct, sigma_fluct_err, sigma_poisson, sigma_obs


def structures_old(data, dz, kind, save=False, path=None):
    xb, y = data_hist(data, dz, bi='edges')
    x = [xb[i] + dz / 2 for i in range(len(xb) - 1)]
    x, y = np.array(x), np.array(y)

    par = ap.fast(x, y, kind)[0]
    delta = (y - y_app(xb, dz, par, kind)) / y_app(xb, dz, par, kind)
    sigma = 1 / y_app(xb, dz, par, kind)

    number, *d = my_main_old(x, delta, sigma)
    table_dict = {'n': d[0],
                  'z start': d[1],
                  'st err com': d[2],
                  'st err pro': d[3],
                  'z finish': d[4],
                  'fin err com': d[5],
                  'fin err pro': d[6],
                  'z middle': d[7],
                  'r comoving': d[8],
                  'r proper': d[9],
                  'sigma fluct': d[10],  # несмещённая дисперсия
                  'sigma fluct err': d[11],
                  'sigma Poisson': d[12],
                  'sigma obs': d[13]
                  }

    table = pd.DataFrame(table_dict, index=number)
    table.index.name = 'j'

    if save:
        if path is not None:
            filename = path + 'dz={:.3f} kind {}.csv'.format(dz, kind)
        else:
            filename = 'dz={:.3f} kind {}.csv'.format(dz, kind)
        table.to_csv(filename)
    else:
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        print(table)


def sigma_dm_func(z, dz):
    sigma_a = 0.069
    sigma_b = 0.234
    betta = 0.834
    return sigma_a / (z ** betta + sigma_b) * np.sqrt(0.2 / dz)


def sigma_pl_func(z, c):
    data = [0.405, 0.310, 0.287, 0.273, 0.263, 0.255, 0.248, 0.242, 0.237, 0.232, 0.228, 0.225, 0.221, 0.218, 0.215,
            0.213, 0.210, 0.208, 0.206, 0.203, 0.201, 0.199, 0.197, 0.196, 0.194, 0.192, 0.190, 0.189, 0.187, 0.186]

    y = [data[i] - data[i + 1] for i in range(len(data) - 1)]
    y_max = max(y)

    y = [i / y_max for i in y]
    y = [log10(i) for i in y]

    x_mid = [0.05 + i * 0.1 for i in range(len(data))]
    x_edge = [0.1 + i * 0.1 for i in range(len(y))]

    p_cont = np.polyfit(x_edge[-c:], y[-c:], 1)

    x_mid_cont = [0.05 + i * 0.1 for i in range(len(data), 60)]
    x_edge_cont = [0.1 + i * 0.1 for i in range(len(y), 59)]
    data_cont = []
    active = data[-1]
    for i in range(len(x_edge_cont)):
        data_cont.append(active - y_max * 10 ** (p_cont[0] * x_edge_cont[i] + p_cont[1]))
        active = data_cont[-1]

    data_full = data + data_cont
    x_mid_full = list(np.around(x_mid + x_mid_cont, 2))
    #
    # t = {'sigma pl': data_full}
    # table = pd.DataFrame(t, index=x_mid_full)
    # table.index.name = 'z'

    # filename = 'sigma pl c=16.csv'
    # table.to_csv(filename)

    sigma_pl = dict(zip(x_mid_full, data_full))
    return sigma_pl[z]


def my_main(z, delta, integrals):
    def start():
        number.append(j + 1)
        n.append(1)
        z_start.append(z[i])
        z_finish.append(z[i])
        z_mid.append(0)
        z_err.append(0)
        r_com.append(0)
        start_err_com.append(0)
        finish_err_com.append(0)
        sigma_poisson.append(integrals[i])
        sigma_obs.append(delta[i])
        sigma_obs_err.append(0)
        sigma_dm.append(0)
        b_dm.append(0)
        sigma_pl.append(0)
        b_pl.append(0)

    def cont():
        n[j] += 1
        sigma_poisson[j] += integrals[i]
        sigma_obs[j] += delta[i]
        sigma_dm[j] += sigma_dm_func(z[i], dz)
        sigma_pl[j] += sigma_pl_func(float(np.around(z[i], 2)), 16)

    def fin():
        z_finish[j] = z[i]
        z_mid[j] = (z_finish[j] + z_start[j]) / 2
        z_err[j] = z_mid[j] - z_start[j]
        r_com[j] = di.r_comoving(z_finish[j]) - di.r_comoving(z_start[j])
        start_err_com[j] = di.r_comoving(z_start[j] + dz) - di.r_comoving(z_start[j] - dz)
        finish_err_com[j] = di.r_comoving(z_finish[j] + dz) - di.r_comoving(z_finish[j] - dz)
        sigma_poisson[j] = np.sqrt(1 / sigma_poisson[j])
        sigma_obs[j] = sigma_obs[j] / n[j]
        sigma_dm[j] = sigma_dm[j] / n[j]
        sigma_pl[j] = sigma_pl[j] / n[j]

        if abs(sigma_obs[j]) <= sigma_poisson[j]:
            b_dm[j] = 0
            b_pl[j] = 0
        else:
            b_dm[j] = np.sqrt(sigma_obs[j] ** 2 - sigma_poisson[j] ** 2) / sigma_dm[j]
            b_pl[j] = np.sqrt(sigma_obs[j] ** 2 - sigma_poisson[j] ** 2) / sigma_pl[j]

        i_start = list(z).index(z_start[j])
        i_finish = list(z).index(z_finish[j])
        for k in range(i_start, i_finish + 1):
            sigma_obs_err[j] += (delta[k] - sigma_obs[j]) ** 2
        sigma_obs_err[j] = np.sqrt(sigma_obs_err[j] / (n[j] - 1)) / np.sqrt(n[j])  #the calculating for standard error of the means

    sgn = []
    i = 0

    if np.sign(delta[i]) == np.sign(delta[i] + delta[i + 1]):
        sgn.append(np.sign(delta[i]))
    else:
        sgn.append(0.0)
    i += 1

    while i < len(delta) - 1:
        if np.sign(delta[i]) == np.sign(delta[i] + delta[i + 1]) == np.sign(delta[i - 1] + delta[i]):
            sgn.append(np.sign(delta[i]))
        else:
            sgn.append(0.0)
        i += 1

    if np.sign(delta[i]) == np.sign(delta[i - 1] + delta[i]):
        sgn.append(np.sign(delta[i]))
    else:
        sgn.append(0.0)

    dz = (z[1] - z[0]) / 2
    number = []
    n = []
    z_start = []
    z_finish = []
    z_mid = []
    z_err = []
    r_com = []
    start_err_com = []
    finish_err_com = []
    sigma_poisson = []
    sigma_obs = []
    sigma_obs_err = []
    sigma_dm = []
    b_dm = []
    sigma_pl = []
    b_pl = []

    i, j = 0, -1
    new = False
    if sgn[0] == 0:
        new = True
    else:
        j += 1
        start()

    for i in range(1, len(sgn) - 1):
        if new:
            if sgn[i + 1] == 0:
                continue

            new = False

            j += 1
            start()

        elif sgn[i] == 0:
            cont()
            fin()

            if sgn[i + 1] == 0:
                new = True
                continue

            j += 1
            start()

        else:
            cont()

    i += 1
    if not new:
        cont()
        fin()

    if z_finish[-1] == z[-1]:
        return number[:-1], n[:-1], z_start[:-1], z_finish[:-1], z_mid[:-1], z_err[:-1], r_com[:-1], \
               start_err_com[:-1], finish_err_com[:-1], sigma_poisson[:-1], sigma_obs[:-1], sigma_obs_err[:-1], \
               sigma_dm[:-1], b_dm[:-1], sigma_pl[:-1], b_pl[:-1]
    else:
        return number, n, z_start, z_finish, z_mid, z_err, r_com, start_err_com, finish_err_com, sigma_poisson, \
               sigma_obs, sigma_obs_err, sigma_dm, b_dm, sigma_pl, b_pl


def structures(data, dz, kind, save=False, path=None):
    xb, y = data_hist(data, dz, bi='edges')
    x = [xb[i] + dz / 2 for i in range(len(xb) - 1)]
    x, y = np.array(x), np.array(y)

    par = ap.fast(x, y, kind)[0]
    delta = (y - y_app(xb, dz, par, kind)) / y_app(xb, dz, par, kind)
    integrals = y_app(xb, dz, par, kind)

    j, *d = my_main(x, delta, integrals)
    t = {'n': d[0] + [''],
         'z middle': np.around(d[3] + [sum(d[3]) / j[-1]], 2),
         'z err': np.around(d[4] + [sum(d[4]) / j[-1]], 2),
         'r comoving': d[5],
         'st err com': d[6],
         'fin err com': d[7],
         'sigma Poisson': d[8] + [sum(d[8]) / j[-1]],
         'sigma obs': d[9] + [sum(d[9]) / j[-1]],
         'sigma obs err': d[10],
         'sigma dm': d[11] + [sum(d[11]) / j[-1]],
         'b dm': d[12],
         'sigma b dm': [],
         'sigma pl': d[13] + [sum(d[13]) / j[-1]],
         'b pl': d[14],
         'sigma b pl': []
         }

    zero_count = 0
    for i in range(j[-1]):
        if t['b dm'][i] == 0:
            zero_count += 1
    t['b dm'] = t['b dm'] + [sum(t['b dm']) / (j[-1] - zero_count)]
    t['b pl'] = t['b pl'] + [sum(t['b pl']) / (j[-1] - zero_count)]

    r_mean = sum(t['r comoving']) / j[-1]
    r_sigma = 0
    sigma_sigma_obs = 0
    sigma_sigma_b_dm = 0
    sigma_sigma_b_pl = 0
    for i in range(j[-1]):
        r_sigma += (t['r comoving'][i] - r_mean) ** 2
        sigma_sigma_obs += (t['sigma obs'][i] - t['sigma obs'][-1]) ** 2
        sigma_sigma_b_dm += (t['b dm'][i] - t['b dm'][-1]) ** 2
        sigma_sigma_b_pl += (t['b pl'][i] - t['b pl'][-1]) ** 2
        t['sigma b dm'].append(t['b dm'][i] * t['sigma obs err'][i] / abs(t['sigma obs'][i]))
        t['sigma b pl'].append(t['b pl'][i] * t['sigma obs err'][i] / abs(t['sigma obs'][i]))

    r_sigma /= j[-1] - 1
    sigma_sigma_obs /= j[-1] - 1
    sigma_sigma_b_dm /= j[-1] - 1
    sigma_sigma_b_pl /= j[-1] - 1

    r_st_err_mean = np.sqrt((sum(t['st err com']) / j[-1]) ** 2 + r_sigma) / np.sqrt(j[-1])       #the calculating for standard error of the means
    r_fin_err_mean = np.sqrt((sum(t['fin err com']) / j[-1]) ** 2 + r_sigma) / np.sqrt(j[-1])     #the calculating for standard error of the means
    sigma_obs_err_mean = np.sqrt(t['sigma Poisson'][-1] ** 2 + sigma_sigma_obs) / np.sqrt(j[-1])  #the calculating for standard error of the means

    sigma_sigma_b_dm /= j[-1]  #the calculating for standard error of the means
    sigma_sigma_b_pl /= j[-1]  #the calculating for standard error of the means

    t['r comoving'] = [int(i) for i in t['r comoving'] + [r_mean]]
    t['st err com'] = [int(i) for i in t['st err com'] + [r_st_err_mean]]
    t['fin err com'] = [int(i) for i in t['fin err com'] + [r_fin_err_mean]]
    t['sigma obs err'] = t['sigma obs err'] + [sigma_obs_err_mean]
    t['b dm'] = np.around(t['b dm'], 2)
    t['sigma b dm'] = np.around(t['sigma b dm'] + [np.sqrt(sigma_sigma_b_dm)], 2)
    t['b pl'] = np.around(t['b pl'], 2)
    t['sigma b pl'] = np.around(t['sigma b pl'] + [np.sqrt(sigma_sigma_b_pl)], 2)

    j += ['means:']
    table = pd.DataFrame(t, index=j)
    table.index.name = 'j'

    if save:
        if path is not None:
            filename = path + 'dz={:.3f} kind {}.csv'.format(dz, kind)
        else:
            filename = 'dz={:.3f} kind {}.csv'.format(dz, kind)
        table.to_csv(filename)
    else:
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        print(table)


def fluct_table(data, dz, kind, save=False, path=None):
    xb, y = data_hist(data, dz, bi='edges')
    x = [xb[i] + dz / 2 for i in range(len(xb) - 1)]
    x, y = np.array(x), np.array(y)

    par = ap.fast(x, y, kind)[0]
    delta = (y - y_app(xb, dz, par, kind)) / y_app(xb, dz, par, kind)
    sigma = 5 / np.sqrt(y_app(xb, dz, par, kind))

    r = [di.r_comoving(xb[i]) - di.r_comoving(xb[i - 1]) for i in range(1, len(xb))]
    d_l = [di.r_proper(xb[i]) - di.r_proper(xb[i - 1]) for i in range(1, len(xb))]

    table_dict = {'z': x,
                  'R': r,
                  'd_L': d_l,
                  'delta': delta,
                  '5 sigma Poisson': sigma
                  }

    table = pd.DataFrame(table_dict)
    table.index.name = 'i'

    if save:
        if path is not None:
            filename = path + 'dz={:.3f} kind {}.csv'.format(dz, kind)
        else:
            filename = 'dz={:.3f} kind {}.csv'.format(dz, kind)
        table.to_csv(filename)
    else:
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        print(table)


def test_sigma_dm():
    z = [i / 10 for i in range(1, 37)]
    dz = [0.05, 0.1, 0.2, 0.3]

    t = {'0.05': [],
         '0.1': [],
         '0.2': [],
         '0.3': []
         }

    for dzi in dz:
        for i in range(len(z)):
            t[str(dzi)].append(sigma_dm_func(z[i], dzi))

    table = pd.DataFrame(t)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    print(table)
