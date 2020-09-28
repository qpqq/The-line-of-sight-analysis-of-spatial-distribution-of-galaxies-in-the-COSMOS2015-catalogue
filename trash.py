import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from math import log10
from astropy.io import fits, ascii
from astropy.table import Table, QTable
from data_processing import z_final, z_med_pdz, z_min_chi2, data_hist, ra_dec, density_plot_data
from distance import r_comoving as rc
from graph import app_graph, fluct_graph, w_hists, r_hists, density_plot, z_vs_z
from PIL import Image


def vs():
    from random import sample

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
            dd = sorted(sample(d, k))
            for t in dd:
                new_data[0].append(t[0])
                new_data[1][0].append(t[1])
                new_data[1][1].append(t[2])

        return new_data

    w = 0.7
    z1 = [[k, 0, 0] for k in z_min_chi2(w=w)]
    z2 = z_final(w=w, err=True)
    z3 = z_med_pdz(w=w, err=True)

    dz = 0.1
    p = 10
    z1 = kk(z1, dz, p)
    z2 = kk(z2, dz, p)
    z3 = kk(z3, dz, p)

    z_vs_z(z2[0], z2[1], z3[0], z3[1])


def f(t, par, kind):
    t = np.array(t)

    if kind == 1:
        alpha, beta, const = par
        return const * t ** alpha * np.exp(-t) ** beta
    elif kind == 2:
        alpha, beta, const = par
        return const * t ** alpha * np.exp(- t / 3) ** beta
    elif kind == 3:
        alpha, beta, const = par
        return const * t ** alpha * np.exp(- t / 6) ** beta


def fast(x, y, kind):
    from scipy.optimize import leastsq

    def diff(parameters):
        return y - f(x, parameters, kind)

    p = np.array([1, 2, 10 ** 5])

    par = leastsq(diff, p, maxfev=10000)[0]

    err = (y - f(x, par, kind)) ** 2
    return par, sum(err)


def z_c_graph(data, func, dz, w, save=False, path=None):
    print('dz={}, w>{}'.format(str(dz), str(w)))

    x, y = data_hist(data, dz, bi='center')

    par1, s1 = fast(x, y, 1)
    par2, s2 = fast(x, y, 2)
    par3, s3 = fast(x, y, 3)

    a1, b1, c1 = par1
    a2, b2, c2 = par2
    a3, b3, c3 = par3

    p = ['{:.2f}'.format(k) for k in [a1, b1, a2, b2, a3, b3]]
    a1, b1, a2, b2, a3, b3 = p
    p = ['{:.0f}'.format(k) for k in [c1, c2, c3]]
    c1, c2, c3 = p

    x0 = np.linspace(x[0], x[-1], 1000)

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.scatter(x, y, s=10, c=[[1, 0, 1] for i in x])
    ax.plot(x, y)

    label1 = r'$\mathrm{' + c1 + 'x^{' + a1 + r'}e^{\left(-\frac{x}{' + '1' + r'}\right)^{' + b1 + '}}}$'
    label1 += ', ' + '%.3e' % s1
    ax.plot(x0, f(x0, par1, 1), color='orange', linestyle='--', linewidth=0.8, label=label1)

    label2 = r'$\mathrm{' + str(int(c2)) + 'x^{' + a2 + r'}e^{\left(-\frac{x}{' + '3' + r'}\right)^{' + b2 + '}}}$'
    label2 += ', ' + '%.3e' % s2
    ax.plot(x0, f(x0, par2, 2), color='limegreen', linestyle='--', linewidth=0.8, label=label2)

    label3 = r'$\mathrm{' + c3 + 'x^{' + a3 + r'}e^{\left(-\frac{x}{' + '6' + r'}\right)^{' + b3 + '}}}$'
    label3 += ', ' + '%.3e' % s3
    ax.plot(x0, f(x0, par3, 3), color='orangered', linestyle='--', linewidth=0.8, label=label3)

    ax.set_xlabel('z', fontsize='x-large')
    ax.set_ylabel('N(z)', fontsize='x-large')

    ax.set_xlim(0)
    ax.set_ylim(0)

    if w == -100:
        plt.title('Approximations for {} with dz={}, all w, N={:d}'.format(func.upper(), str(dz), sum(y)),
                  fontsize='x-large')
    else:
        plt.title('Approximations for {} with dz={}, w>{}, N={:d}'.format(func.upper(), str(dz), str(w), sum(y)),
                  fontsize='x-large')

    ax.legend(fontsize='x-large')
    ax.tick_params(labelsize='x-large')

    if save:
        if path is not None:
            filename = path + 'dz={:.3f}.png'.format(dz)
        else:
            filename = 'dz={:.3f}.png'.format(dz)
        plt.savefig(filename, dpi=150)
    else:
        plt.show()

    plt.close()


def z_c_fluct(data, func, dz, w, save=False, path=None):
    from matplotlib.ticker import FixedLocator
    from scipy import integrate

    def y_app(bins, par, kind):
        yapp = []
        for i in range(1, len(bins)):
            yapp.append(integrate.quad(f, bins[i - 1], bins[i], args=(par, kind))[0] / dz)
        return np.array(yapp)

    print('dz={}, w>{}'.format(dz, w))

    xb, y = data_hist(data, dz, bi='edges')
    x = [xb[i] + dz / 2 for i in range(len(xb) - 1)]
    x, y = np.array(x), np.array(y)

    par1, s1 = fast(x, y, 1)
    par2, s2 = fast(x, y, 2)
    par3, s3 = fast(x, y, 3)

    a1, b1, c1 = par1
    a2, b2, c2 = par2
    a3, b3, c3 = par3

    p = ['{:.2f}'.format(k) for k in [a1, b1, a2, b2, a3, b3]]
    a1, b1, a2, b2, a3, b3 = p
    p = ['{:.0f}'.format(k) for k in [c1, c2, c3]]
    c1, c2, c3 = p

    ig, ax = plt.subplots(figsize=(12.8, 7.2))

    label1 = r'$\mathrm{' + c1 + 'x^{' + a1 + r'}e^{\left(-\frac{x}{' + '1' + r'}\right)^{' + b1 + '}}}$'
    label1 += ', ' + '%.3e' % s1
    ax.plot(x, (y - y_app(xb, par1, 1)) / y_app(xb, par1, 1), color='orange', linewidth=0.9, label=label1)
    ax.plot(x, 5 / np.sqrt(y_app(xb, par1, 1)), color='orange', linestyle='--', linewidth=0.5)
    ax.plot(x, -5 / np.sqrt(y_app(xb, par1, 1)), color='orange', linestyle='--', linewidth=0.5)

    label2 = r'$\mathrm{' + c2 + 'x^{' + a2 + r'}e^{\left(-\frac{x}{' + '3' + r'}\right)^{' + b2 + '}}}$'
    label2 += ', ' + '%.3e' % s2
    ax.plot(x, (y - y_app(xb, par2, 2)) / y_app(xb, par2, 2), color='limegreen', linewidth=0.9, label=label2)
    ax.plot(x, 5 / np.sqrt(y_app(xb, par2, 2)), color='limegreen', linestyle='--', linewidth=0.5)
    ax.plot(x, -5 / np.sqrt(y_app(xb, par2, 2)), color='limegreen', linestyle='--', linewidth=0.5)

    label3 = r'$\mathrm{' + c3 + 'x^{' + a3 + r'}e^{\left(-\frac{x}{' + '6' + r'}\right)^{' + b3 + '}}}$'
    label3 += ', ' + '%.3e' % s3
    ax.plot(x, (y - y_app(xb, par3, 3)) / y_app(xb, par3, 3), color='orangered', linewidth=0.9, label=label3)
    ax.plot(x, 5 / np.sqrt(y_app(xb, par3, 3)), color='orangered', linestyle='--', linewidth=0.5)
    ax.plot(x, -5 / np.sqrt(y_app(xb, par3, 3)), color='orangered', linestyle='--', linewidth=0.5)

    ax.tick_params(axis='x', direction='inout')
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 3, 4, 5, 6]))
    ax.tick_params(labelsize='x-large')

    ax.spines['bottom'].set_position(('data', 0.0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('z', fontsize='x-large')
    ax.xaxis.set_label_coords(1.01, 0.516)
    ax.set_ylabel(r'$\mathrm{\delta=\frac{\Delta N_{obs} - \Delta N_{approx}}{\Delta N_{approx}}}$',
                  fontsize='xx-large')

    ax.set_xlim(0, 6.1)

    if w != 'all w' and float(w) > 0.8:
        ax.set_ylim(-1, 1)
    else:
        limits = plt.axis()
        ylim = max(abs(limits[2]), abs(limits[3]))
        ax.set_ylim(-ylim, ylim)

    if w == -100:
        plt.title('Fluctuations for {} with dz={}, all w, N={:d}'.format(func.upper(), dz, sum(y)),
                  fontsize='x-large')
    else:
        plt.title('Fluctuations for {} with dz={}, w>{}, N={:d}'.format(func.upper(), dz, w, sum(y)),
                  fontsize='x-large')

    ax.legend(fontsize='x-large')

    if save:
        if path is not None:
            filename = path + 'dz={:.3f}.png'.format(dz)
        else:
            filename = 'dz={:.3f}.png'.format(dz)
        plt.savefig(filename, dpi=150)
    else:
        plt.show()

    plt.close()


def many_w_hists(dz, w, path, file=None):
    """
    many_w_hists([0.3, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.005], [-100, 0.7, 0.8, 0.9, 0.97], 'histograms')
    """

    try:
        home = os.getcwd()
        os.makedirs(f'{home}/{path}')
    except FileExistsError:
        pass

    path += '/'
    im = []
    for dzi in dz:
        w_hists(dzi, w, save=True, path=path, file=file)

        im.append(Image.open(f'{path}dz={dzi:.3f}.png'))

    im[0].save(f'{path}.3.2.1.08.06.04.02.005.gif', save_all=True, append_images=im[1:], duration=1200, loop=0)


def many_r_hists(dr, w, path, file=None):
    """
    many_r_hists([1200, 800, 400, 300, 200, 100, 50], [-100, 0.7, 0.8, 0.9, 0.97], 'histograms r')
    """

    try:
        home = os.getcwd()
        os.makedirs(f'{home}/{path}')
    except FileExistsError:
        pass

    path += '/'
    im = []
    for dri in dr:
        r_hists(dri, w, save=True, path=path)

        im.append(Image.open(f'{path}dr={dri}.png'))

    im[0].save(f'{path}.120.80.40.30.20.10.5.gif', save_all=True, append_images=im[1:], duration=1200, loop=0)


def fits_1():
    """
    COSMOS2015v1.1:         length = 1182108
    pdz_cosmos2015_v1.3_t1: length =  606887
    FLAG_COSMOS:            length =  576762
    FLAG_HJMCC:             length =  606887
    FLAG_DEEP:              length =  227278
    FLAG_PETER:             length =  539706
    """

    data_full = fits.open('COSMOS2015v1.1.fits')[1].data
    data_phot = Table.read('pdz_cosmos2015_v1.3_t1.csv', format='ascii.csv')

    # print(data['FLAG_HJMCC'])
    # print(data['FLAG_DEEP'])
    # print(data['FLAG_COSMOS'])
    # print(data['FLAG_PETER'])
    # print(data['number'])

    ind_set = set()
    ind_dict = {}
    for i, ind in enumerate(data_phot['ID']):
        ind_set.add(ind)
        ind_dict[ind] = i

    max_ind = 1182108
    subset = 'FLAG_PETER'

    names = data_phot.colnames
    values = [[] for i in names]
    table = QTable(values, names=data_phot.colnames)
    for i in range(max_ind):
        if data_full[subset][i] == 0 and (i + 1) in ind_set:
            ind = ind_dict[i + 1]
            print(str(i + 1)[:2], end=' ')
            table.add_row(data_phot[ind])

    ascii.write(table, f'{subset}.csv', format='csv')


def fits_2():
    data = fits.open('COSMOS2015v1.1.fits')[1].data

    max_ind = 1182108
    ind = {'uv_3': [], 'uv_6': [], 'c_3': [], 'c_6': []}
    z = {'uv_3': [], 'uv_6': [], 'c_3': [], 'c_6': []}
    dz_l = {'uv_3': [], 'uv_6': [], 'c_3': [], 'c_6': []}
    dz_h = {'uv_3': [], 'uv_6': [], 'c_3': [], 'c_6': []}

    for i in range(max_ind):
        # print(i, end=' ')
        z_i = data['ZPDF'][i]

        if z_i < 0:
            continue

        '''
        if data['FLAG_HJMCC'][i] == 0:
            if data['FLAG_DEEP'][i] == 0 and z_i < 3:
                ind['uv_3'].append(i + 1)
                z['uv_3'].append(z_i)
                dz_l['uv_3'].append(data['ZPDF_L68'][i])
                dz_h['uv_3'].append(data['ZPDF_H68'][i])

            if data['FLAG_DEEP'][i] == 1 and z_i < 6:
                ind['uv_6'].append(i + 1)
                z['uv_6'].append(z_i)
                dz_l['uv_6'].append(data['ZPDF_L68'][i])
                dz_h['uv_6'].append(data['ZPDF_H68'][i])

        if data['FLAG_COSMOS'][i] == 1:
            if z_i < 3:
                ind['c_3'].append(i + 1)
                z['c_3'].append(z_i)
                dz_l['c_3'].append(data['ZPDF_L68'][i])
                dz_h['c_3'].append(data['ZPDF_H68'][i])

            if z_i < 6:
                ind['c_6'].append(i + 1)
                z['c_6'].append(z_i)
                dz_l['c_6'].append(data['ZPDF_L68'][i])
                dz_h['c_6'].append(data['ZPDF_H68'][i])
        
        
        if data['FLAG_HJMCC'][i] == 0 and data['FLAG_COSMOS'][i] != 1 and data['FLAG_DEEP'][i] != 1 and z_i < 3:
            ind['uv_3'].append(i + 1)
            z['uv_3'].append(z_i)
            dz_l['uv_3'].append(data['ZPDF_L68'][i])
            dz_h['uv_3'].append(data['ZPDF_H68'][i])

        if data['FLAG_COSMOS'][i] == 1 and data['FLAG_HJMCC'][i] != 0 and data['FLAG_DEEP'][i] != 1 and z_i < 3:
            ind['c_3'].append(i + 1)
            z['c_3'].append(z_i)
            dz_l['c_3'].append(data['ZPDF_L68'][i])
            dz_h['c_3'].append(data['ZPDF_H68'][i])
        

        if data['FLAG_COSMOS'][i] != 1 and data['FLAG_HJMCC'][i] != 0 and data['FLAG_DEEP'][i] == 1 and z_i < 6:
            ind['uv_6'].append(i + 1)
            z['uv_6'].append(z_i)
            dz_l['uv_6'].append(data['ZPDF_L68'][i])
            dz_h['uv_6'].append(data['ZPDF_H68'][i])
            

        if data['FLAG_COSMOS'][i] != 1 and data['FLAG_HJMCC'][i] == 0 and data['FLAG_DEEP'][i] == 1 and z_i < 6:
            ind['uv_6'].append(i + 1)
            z['uv_6'].append(z_i)
            dz_l['uv_6'].append(data['ZPDF_L68'][i])
            dz_h['uv_6'].append(data['ZPDF_H68'][i])
        '''

        if data['FLAG_COSMOS'][i] != 1 and (data['FLAG_HJMCC'][i] == 0 or data['FLAG_DEEP'][i] == 1) and z_i < 3:
            ind['c_3'].append(i + 1)
            z['c_3'].append(z_i)
            dz_l['c_3'].append(data['ZPDF_L68'][i])
            dz_h['c_3'].append(data['ZPDF_H68'][i])

    print()

    names = ['ID', 'ZPDF', 'ZPDF_L68', 'ZPDF_H68']
    # ascii.write([ind['uv_3'], z['uv_3'], dz_l['uv_3'], dz_h['uv_3']], 'UltraVISTA 3 without others.csv', names=names, format='csv')
    # ascii.write([ind['uv_6'], z['uv_6'], dz_l['uv_6'], dz_h['uv_6']], 'UV + deep without COSMOS.csv', names=names, format='csv')
    ascii.write([ind['c_3'], z['c_3'], dz_l['c_3'], dz_h['c_3']], 'UV + deep 3 without COSMOS.csv', names=names,
                format='csv')
    # ascii.write([ind['c_6'], z['c_6'], dz_l['c_6'], dz_h['c_6']], 'COSMOS 6.csv', names=names, format='csv')


def main(dz, w, mode, path='', file=None, corr=False):
    """
    dz = [0.3, 0.2, 0.1, 0.075, 0.05, 0.03]
    """
    from table import fluct_table, structures

    home = os.getcwd()

    for wi in w:
        try:
            if wi == -100:
                os.makedirs(f'{home}/{path}{mode}/all w')
            else:
                os.makedirs(f'{home}/{path}{mode}/w={wi}')
        except FileExistsError:
            continue

    for wi in w:
        for dzi in dz:
            if corr:
                data = Table.read(f'{file}.csv', format='ascii.csv')['ZPDF']
                # data = Table.read(f'{file}.csv', format='ascii.csv')['R']
                data = sorted(data)
            elif file:
                data = z_final(file=file.upper(), w=wi)
            else:
                data = z_final(w=wi)

            if wi == -100:
                path_i = f'{path}{mode}/all w/'
            else:
                path_i = f'{path}{mode}/w={wi}/'

            if mode == 'approx':
                app_graph(data, 'z_final', dzi, wi, save=True, path=path_i)
            elif mode == 'fluctuations':
                fluct_graph(data, 'z_final', dzi, wi, save=True, path=path_i)
            elif mode == 'fluctuations tables':
                print(f'dz={dzi}, w>{wi}')
                fluct_table(data, dzi, 1, save=True, path=path_i)
                fluct_table(data, dzi, 2, save=True, path=path_i)
                fluct_table(data, dzi, 3, save=True, path=path_i)
            elif mode == 'structures':
                print(f'dz={dzi}, w>{wi}')
                structures(data, dzi, 1, save=True, path=path_i)
                structures(data, dzi, 2, save=True, path=path_i)
                structures(data, dzi, 3, save=True, path=path_i)


def main_all(dz, w, file=None, corr=False):
    mode_list = ['approx', 'fluctuations', 'fluctuations tables', 'structures']

    for mode in mode_list:
        print(mode)

        if file:
            main(dz, w, mode, f'{file}/', file, corr)
        else:
            main(dz, w, mode)


def correlation(path_1, path_2, save=False):
    dz = [0.3, 0.2, 0.1, 0.075, 0.05, 0.03]

    ro = [[], [], []]
    sigma = [[], [], []]
    for dzi in dz:
        for kind in range(1, 4):
            data_10 = Table.read(f'{path_1}dz={dzi:.3f} kind {kind}.csv', format='ascii.csv')
            data_20 = Table.read(f'{path_2}dz={dzi:.3f} kind {kind}.csv', format='ascii.csv')

            data_1 = []
            data_2 = []
            for i in range(len(data_10['z'])):
                if data_10['z'][i] < 6:
                    data_1.append(data_10['delta'][i])
                if data_20['z'][i] < 6:
                    data_2.append(data_20['delta'][i])

            data_1 = np.array(data_1)
            data_2 = np.array(data_2)

            roi = (data_1 - np.average(data_1)) * (data_2 - np.average(data_2))
            roi = sum(roi)
            roi /= np.sqrt(sum((data_1 - np.average(data_1)) ** 2) * sum((data_2 - np.average(data_2)) ** 2))
            ro[kind - 1].append(roi)

            sigmai = np.sqrt(1 - roi ** 2) / np.sqrt(len(data_1) - 2)
            sigma[kind - 1].append(sigmai)

    names = ['dz i', 'ro 1', 'sigma 1', 'ro 2', 'sigma 2', 'ro 3', 'sigma 3']
    data = [dz, ro[0], sigma[0], ro[1], sigma[1], ro[2], sigma[2]]
    if save:
        ascii.write(data, 'result 6(3).csv', names=names, format='csv')
    else:
        table_dict = dict(zip(names, data))
        table = pd.DataFrame(table_dict)

        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        print(table)


def z_to_r(file):
    data = Table.read(f'{file}.csv', format='ascii.csv')

    r = []
    for z in data['ZPDF']:
        r.append(rc(z))

    data['R'] = r

    data.write(f'{file} r.csv', format='csv')


def f1(save=False):
    y = [0.405, 0.310, 0.287, 0.273, 0.263, 0.255, 0.248, 0.242, 0.237, 0.232, 0.228, 0.225, 0.221, 0.218, 0.215, 0.213,
         0.210, 0.208, 0.206, 0.203, 0.201, 0.199, 0.197, 0.196, 0.194, 0.192, 0.190, 0.189, 0.187, 0.186]
    x = [0.05 + i * 0.1 for i in range(len(y))]

    c = 2
    z = np.polyfit(x[-c:], y[-c:], 1)
    f = np.poly1d(z)

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.scatter(x, y, s=20)

    xx = np.linspace(0, 6, 1000)
    ax.plot(xx, f(xx), color='orangered', linestyle='--', linewidth=1)
    ax.plot(xx, [y[-1] for i in range(len(xx))], color='grey', linestyle='--', linewidth=0.6, alpha=0.6)

    ax.set_xlim(-0.05, 6.05)
    ax.set_ylim(0, 0.45)

    ax.set_xlabel('z')
    ax.set_ylabel('sigma pl')
    plt.title(f'Просто точки (точек аппроксимации: {c})')

    if save:
        filename = f'точки {c}.png'
        plt.savefig(filename, dpi=150)
    else:
        plt.show()


def f2(save=False):
    data = [0.405, 0.310, 0.287, 0.273, 0.263, 0.255, 0.248, 0.242, 0.237, 0.232, 0.228, 0.225, 0.221, 0.218, 0.215,
            0.213, 0.210, 0.208, 0.206, 0.203, 0.201, 0.199, 0.197, 0.196, 0.194, 0.192, 0.190, 0.189, 0.187, 0.186]

    y = [data[i] - data[i + 1] for i in range(len(data) - 1)]
    x = [0.1 + i * 0.1 for i in range(len(y))]

    c = 7
    z = np.polyfit(x[-c:], y[-c:], 1)
    f = np.poly1d(z)

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.scatter(x, y, s=20)

    xx = np.linspace(0, 6, 1000)
    ax.plot(xx, f(xx), color='orangered', linestyle='--', linewidth=1)
    ax.plot(xx, [y[-1] for i in range(len(xx))], color='grey', linestyle='--', linewidth=0.6, alpha=0.6)

    ax.set_xlim(-0.1, 6.1)
    ax.set_ylim(-0.002, 0.1)

    ax.set_xlabel('z')
    ax.set_ylabel('delta sigma pl')
    plt.title(f'Разность (точек аппроксимации: {c})')

    if save:
        filename = f'разность {c}.png'
        plt.savefig(filename, dpi=150)
    else:
        plt.show()


def f3(save=False):
    data = [0.405, 0.310, 0.287, 0.273, 0.263, 0.255, 0.248, 0.242, 0.237, 0.232, 0.228, 0.225, 0.221, 0.218, 0.215,
            0.213, 0.210, 0.208, 0.206, 0.203, 0.201, 0.199, 0.197, 0.196, 0.194, 0.192, 0.190, 0.189, 0.187, 0.186]

    y = [data[i] - data[i + 1] for i in range(len(data) - 1)]
    y = [i / max(y) for i in y]
    x = [0.1 + i * 0.1 for i in range(len(y))]

    c = 7
    z = np.polyfit(x[-c:], y[-c:], 1)
    f = np.poly1d(z)

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.scatter(x, y, s=20)

    xx = np.linspace(0, 6, 1000)
    ax.plot(xx, f(xx), color='orangered', linestyle='--', linewidth=1)
    ax.plot(xx, [y[-1] for i in range(len(xx))], color='grey', linestyle='--', linewidth=0.6, alpha=0.6)

    ax.set_xlim(-0.1, 6.1)
    # ax.set_ylim(-0.05, 1.05)

    ax.set_xlabel('z')
    ax.set_ylabel('norm delta sigma pl')
    plt.title(f'Нормированная разность (точек аппроксимации: {c})')

    if save:
        filename = f'норм разность {c}.png'
        plt.savefig(filename, dpi=150)
    else:
        plt.show()


def f4(save=False):
    data = [0.405, 0.310, 0.287, 0.273, 0.263, 0.255, 0.248, 0.242, 0.237, 0.232, 0.228, 0.225, 0.221, 0.218, 0.215,
            0.213, 0.210, 0.208, 0.206, 0.203, 0.201, 0.199, 0.197, 0.196, 0.194, 0.192, 0.190, 0.189, 0.187, 0.186]

    y = [data[i] - data[i + 1] for i in range(len(data) - 1)]
    y = [i / max(y) for i in y]
    y = [log10(i) for i in y]
    x = [0.1 + i * 0.1 for i in range(len(y))]

    c = 7
    z = np.polyfit(x[-c:], y[-c:], 1)
    f = np.poly1d(z)

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.scatter(x, y, s=20)

    xx = np.linspace(0, 6, 1000)
    ax.plot(xx, f(xx), color='orangered', linestyle='--', linewidth=1)
    ax.plot(xx, [y[-1] for i in range(len(xx))], color='grey', linestyle='--', linewidth=0.6, alpha=0.6)

    ax.set_xlim(-0.1, 6.1)
    # ax.set_ylim(-4.8, 0.1)

    ax.set_xlabel('z')
    ax.set_ylabel('log norm delta sigma pl')
    plt.title(f'Логарифмическая нормированная разность (точек аппроксимации: {c})')

    if save:
        filename = f'лог норм разность {c}.png'
        plt.savefig(filename, dpi=150)
    else:
        plt.show()


def f5(c, save=False):
    data = [0.405, 0.310, 0.287, 0.273, 0.263, 0.255, 0.248, 0.242, 0.237, 0.232, 0.228, 0.225, 0.221, 0.218, 0.215,
            0.213, 0.210, 0.208, 0.206, 0.203, 0.201, 0.199, 0.197, 0.196, 0.194, 0.192, 0.190, 0.189, 0.187, 0.186]

    y = [data[i] - data[i + 1] for i in range(len(data) - 1)]
    y_max = max(y)

    y = [i / y_max for i in y]
    y = [log10(i) for i in y]

    x_mid = [0.05 + i * 0.1 for i in range(len(data))]
    x_edge = [0.1 + i * 0.1 for i in range(len(y))]

    p = np.polyfit(x_mid[-2:], data[-2:], 1)

    p_cont = np.polyfit(x_edge[-c:], y[-c:], 1)

    x_mid_cont = [0.05 + i * 0.1 for i in range(len(data), 60)]
    x_edge_cont = [0.1 + i * 0.1 for i in range(len(y), 59)]
    data_cont = []
    active = data[-1]
    for i in range(len(x_edge_cont)):
        data_cont.append(active - y_max * 10 ** (p_cont[0] * x_edge_cont[i] + p_cont[1]))
        active = data_cont[-1]

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.scatter(x_mid, data, s=20)
    ax.scatter(x_mid_cont, data_cont, s=20, c='red')

    x_line = np.linspace(0, 6, 1000)
    ax.plot(x_line, [p[0] * i + p[1] for i in x_line], color='orangered', linestyle='--', linewidth=1)
    ax.plot(x_line, [data[-1] for i in x_line], color='grey', linestyle='--', linewidth=0.6, alpha=0.6)

    ax.set_xlim(-0.1, 6.1)
    # ax.set_ylim(-4.8, 0.1)

    ax.set_xlabel('z')
    ax.set_ylabel('sigma pl')
    plt.title(f'Итого (точек аппроксимации: {c})')

    if save:
        filename = f'итог {c}.png'
        plt.savefig(filename, dpi=150)
    else:
        plt.show()


def slices(start, finish, delta, step, p, w, norm, path=None):
    if path is None:
        path = f'slices from z={start} to z={finish}, delta={delta}, step={step}, p={p} arcmin, w={w}, '

        if norm:
            path += 'norm'
        else:
            path += 'ne norm'

    try:
        home = os.getcwd()
        os.makedirs(f'{home}/{path}')
    except FileExistsError:
        pass

    path += '/'

    data, n_ra, n_dec = density_plot_data(start, finish, delta, step, p, w)

    if norm:
        density_max = -1
        for i in range(len(data)):
            density_data = data[i][1]

            for j in range(n_dec):
                density_max = max(max(density_data[j]), density_max)
    else:
        density_max = None

    i = 0
    im = []
    cur_st = start
    cur_fin = np.around(start + delta, 3)
    while cur_fin <= finish:
        print(cur_st, cur_fin)

        ra_dec_data, density_data = data[i]

        if w == -100:
            title = f'Slice from z={cur_st} to z={cur_fin}, all w, N={len(ra_dec_data[0])}'
        else:
            title = f'Slice from z={cur_st} to z={cur_fin}, w={w}, N={len(ra_dec_data[0])}'

        density_plot(ra_dec_data, density_data, density_max, n_ra, n_dec, p, title=title, save=True, path=path)

        im.append(Image.open(f'{path}{title}.png'))

        i += 1
        cur_st = np.around(cur_st + step, 3)
        cur_fin = np.around(cur_fin + step, 3)

    im[0].save(f'{path}Slices from z={start} to z={finish}, delta={delta}, step={step}, w={w}.gif',
               save_all=True, append_images=im[1:], duration=500, loop=0)  # duration=500


# correlation('COSMOS 6 without others/fluctuations tables/all w/', 'UV + deep 6 without COSMOS/fluctuations tables/all w/', save=True)
# fits_2()
# main_all([0.3, 0.2, 0.1, 0.075, 0.05, 0.03], [-100, 0.7, 0.8, 0.9, 0.97], 'UV + deep 6 without COSMOS', corr=True)
# main([0.3, 0.2, 0.1, 0.075, 0.05, 0.03], [-100, 0.7, 0.8, 0.9, 0.97], 'approx', '', 'UV + deep 3 without COSMOS', True)
# main([0.3, 0.2, 0.1, 0.075, 0.05, 0.03], [-100, 0.7, 0.8, 0.9, 0.97], 'fluctuations', 'COSMOS 3 without others/', 'COSMOS 3 without others', True)
# # dr = [rc(i) for i in [0.3, 0.2, 0.1, 0.075, 0.05, 0.03]]
# main_all([200], [-100], 'UV + deep 6 without COSMOS r', corr=True)
# z_to_r('UV + deep 6 without COSMOS')
# main([200], [-100], 'approx', 'UV + deep 6 without COSMOS r/', 'UV + deep 6 without COSMOS r', True)
# main([0.1], [-100, 0.7, 0.8, 0.9, 0.97], 'structures')
# many_w_hists([0.3, 0.2, 0.1, 0.075, 0.05, 0.03], [-100, 0.7, 0.9, 0.97], 'histograms')
# many_r_hists([1200, 800, 400, 300, 200, 100, 50], [-100, 0.7, 0.9, 0.97], 'histograms r')
# slices(0, 2, delta=0.2, step=0.05, w=0.9, p=2, norm=True)
# slices(1, 2, delta=0.2, step=0.05, w=0.9, p=2, norm=False)
# density_plot_data(2, 3, delta=0.1, step=0.02, w=0.9, n=50)

'''
main([0.1], [-100, 0.7, 0.9, 0.97], 'structures', '123/')
Table 2, 3

main([0.1], [-100], 'structures', 'COSMOS 3 without others/', 'COSMOS 3 without others', True)
main([0.1], [-100], 'structures', 'COSMOS 6 without others/', 'COSMOS 6 without others', True)
main([0.1], [-100], 'structures', 'UV + deep 3 without COSMOS/', 'UV + deep 3 without COSMOS', True)
main([0.1], [-100], 'structures', 'UV + deep 6 without COSMOS/', 'UV + deep 6 without COSMOS', True)
Table A1, A2
'''
