"""
Fractal Dimension Estimation
© Stanislav Shirokov, 2014-2020
"""

import approx as ap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms
import mpl_toolkits.axes_grid1.axes_size as Size
from data_processing import data_hist
from mpl_toolkits.axes_grid1.axes_divider import HBoxDivider


frac = 2 / 3


def app_graph(data, func, dz, w, save=False, path=None):
    print(f'dz={dz}, w>{w}')

    x, y = data_hist(data, dz, bi='center')

    par1, s1 = ap.fast(x, y, 1)
    par2, s2 = ap.fast(x, y, 2)
    par3, s3 = ap.fast(x, y, 3)

    a1, b1, c1 = par1
    a2, b2, z2, c2 = par2
    a3, b3, d3, c3 = par3

    p = ['{:.2f}'.format(k) for k in [a1, b1, a2, b2, z2, a3, b3, d3]]
    a1, b1, a2, b2, z2, a3, b3, d3 = p

    x0 = np.linspace(x[0], x[-1], 1000)

    fig, ax = plt.subplots(figsize=(12.8 * frac, 7.2 * frac))
    ax.scatter(x, y, s=10, c=[[1, 0, 1] for i in x])
    ax.plot(x, y)

    label1 = r'$\mathrm{' + str(int(c1)) + 'z^{' + a1 + '}e^{-' + b1 + 'z}}$'
    label1 += ', ' + '%.3e' % s1
    ax.plot(x0, ap.f(x0, par1, 1), color='orange', linestyle='--', linewidth=0.8, label=label1)

    label2 = r'$\mathrm{' + str(int(c2)) + 'z^{' + a2 + r'}e^{\left(-\frac{z}{' + z2 + r'}\right)^{' + b2 + '}}}$'
    label2 += ', ' + '%.3e' % s2
    ax.plot(x0, ap.f(x0, par2, 2), color='limegreen', linestyle='--', linewidth=0.8, label=label2)

    label3 = r'$\mathrm{' + str(int(c3)) + r'\left(\frac{z^{' + a3 + '}+z^{' + a3 + r'\cdot' + b3 + '}}{z^{' + b3 + \
             '}+' + d3 + r'}\right)}$'
    label3 += ', ' + '%.3e' % s3
    ax.plot(x0, ap.f(x0, par3, 3), color='orangered', linestyle='--', linewidth=0.8, label=label3)

    ax.set_xlabel('z', fontsize='x-large')
    ax.set_ylabel(r'$\mathrm{\Delta N(z,\Delta z)}$', fontsize='x-large')

    ax.set_xlim(0)
    ax.set_ylim(0)

    if w == -100:
        plt.title('Approximations with dz={}, all w, N={:d}'.format(str(dz), sum(y)),
                  fontsize='x-large')
    else:
        plt.title('Approximations with dz={}, w>{}, N={:d}'.format(str(dz), str(w), sum(y)),
                  fontsize='x-large')

    ax.legend(fontsize='x-large')

    # if the legend is superimposed on the graph
    # ax.legend(fontsize='x-large', loc='lower left', bbox_to_anchor=(0.05, 0))

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


def fluct_graph(data, func, dz, w, save=False, path=None):
    from matplotlib.ticker import FixedLocator
    from scipy import integrate

    def y_app(bins, par, kind):
        yapp = []
        for i in range(1, len(bins)):
            yapp.append(integrate.quad(ap.f, bins[i - 1], bins[i], args=(par, kind))[0] / dz)
        return np.array(yapp)

    print(f'dz={dz}, w>{w}')

    xb, y = data_hist(data, dz, bi='edges')
    x = [xb[i] + dz / 2 for i in range(len(xb) - 1)]
    x, y = np.array(x), np.array(y)

    par1, s1 = ap.fast(x, y, 1)
    par2, s2 = ap.fast(x, y, 2)
    par3, s3 = ap.fast(x, y, 3)

    a1, b1, c1 = par1
    a2, b2, z2, c2 = par2
    a3, b3, d3, c3 = par3

    p = ['{:.2f}'.format(k) for k in [a1, b1, a2, b2, z2, a3, b3, d3]]
    a1, b1, a2, b2, z2, a3, b3, d3 = p
    p = ['{:.0f}'.format(k) for k in [c1, c2, c3]]
    c1, c2, c3 = p

    ig, ax = plt.subplots(figsize=(12.8 * frac, 7.2 * frac))

    # label_sigma = r'$5\sigma$'

    label1 = r'$\mathrm{' + c1 + 'z^{' + a1 + '}e^{-' + b1 + 'z}}$'
    label1 += ', ' + '%.3e' % s1
    ax.plot(x, (y - y_app(xb, par1, 1)) / y_app(xb, par1, 1), color='orange', linewidth=0.9, label=label1)
    ax.plot(x, 5 / np.sqrt(y_app(xb, par1, 1)), color='orange', linestyle='--', linewidth=0.5)
    ax.plot(x, -5 / np.sqrt(y_app(xb, par1, 1)), color='orange', linestyle='--', linewidth=0.5)

    label2 = r'$\mathrm{' + c2 + 'z^{' + a2 + r'}e^{\left(-\frac{z}{' + z2 + r'}\right)^{' + b2 + '}}}$'
    label2 += ', ' + '%.3e' % s2
    ax.plot(x, (y - y_app(xb, par2, 2)) / y_app(xb, par2, 2), color='limegreen', linewidth=0.9, label=label2)
    ax.plot(x, 5 / np.sqrt(y_app(xb, par2, 2)), color='limegreen', linestyle='--', linewidth=0.5)
    ax.plot(x, -5 / np.sqrt(y_app(xb, par2, 2)), color='limegreen', linestyle='--', linewidth=0.5)

    label3 = r'$\mathrm{' + c3 + r'\left(\frac{z^{' + a3 + '}+z^{' + a3 + r'\cdot' + b3 + '}}{z^{' + b3 + '}+' + d3 + \
             r'}\right)}$'
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

    # if the name of the x axis is superimposed on numbers
    if frac < 1:
        ax.set_ylabel(r'$\mathrm{\delta=\frac{\Delta N_{obs} - \Delta N_{approx}}{\Delta N_{approx}}}$',
                      fontsize='xx-large', labelpad=-10)
    else:
        ax.set_ylabel(r'$\mathrm{\delta=\frac{\Delta N_{obs} - \Delta N_{approx}}{\Delta N_{approx}}}$',
                      fontsize='xx-large')

    ax.set_xlim(0)

    if float(w) > 0.8:
        ax.set_ylim(-1.05, 1.05)
    else:
        limits = plt.axis()
        ylim = max(abs(limits[2]), abs(limits[3]))
        ax.set_ylim(-ylim, ylim)

    if w == -100:
        plt.title('Fluctuations with dz={}, all w, N={:d}'.format(dz, sum(y)),
                  fontsize='x-large')
    else:
        plt.title('Fluctuations with dz={}, w>{}, N={:d}'.format(dz, w, sum(y)),
                  fontsize='x-large')

    ax.legend(fontsize='x-large')

    # if the legend is superimposed on the graph
    # ax.legend(fontsize='x-large', loc='lower left', bbox_to_anchor=(0.02, 0))
    # ax.set_ylim(-1.6, 1.6)

    if save:
        if path is not None:
            filename = path + 'dz={:.3f}.png'.format(dz)
        else:
            filename = 'dz={:.3f}.png'.format(dz)
        plt.savefig(filename, dpi=150)
    else:
        plt.show()

    plt.close()


def w_hists(dz, w, integral=False, save=False, path=None, file=None):
    from data_processing import z_final
    # from data_processing import z_med_pdz

    print('dz={}'.format(dz))

    x = []
    y = []
    y_sum = []
    for wi in w:
        data = z_final(w=wi, file=file)
        # data = z_med_pdz(w=wi)
        xx, yy = data_hist(data, dz)

        if integral:
            yyy = [yy[0]]
            for i in range(1, len(yy)):
                yyy.append(yyy[-1] + yy[i])

            y.append(yyy)
            y_sum.append(yyy[-1])
        else:
            y.append(yy)
            y_sum.append(sum(yy))

        x.append(xx)

    fig, ax = plt.subplots(figsize=(12.8 * frac, 7.2 * frac))

    if w[0] == -100:
        label = 'all w, N={:d}'.format(y_sum[0])
        ax.hist(x[0][:-1], bins=x[0], weights=y[0], histtype='step', label=label)
    else:
        label = 'w>{}, N={:d}'.format(str(w[0]), y_sum[0])
        ax.hist(x[0][:-1], bins=x[0], weights=y[0], histtype='stepfilled', alpha=0.9, label=label)

    for i in range(1, len(w)):
        label = 'w>{}, N={:d}'.format(str(w[i]), y_sum[i])
        if i == 4:
            ax.hist(x[i][:-1], bins=x[i], weights=y[i], histtype='stepfilled', color='#FFD94A', alpha=0.9, label=label)
        else:
            ax.hist(x[i][:-1], bins=x[i], weights=y[i], histtype='stepfilled', alpha=0.9, label=label)

    ax.set_xlabel('z', fontsize='x-large')
    ax.set_ylabel(r'$\mathrm{\Delta N(z,\Delta z)}$', fontsize='x-large')

    ax.set_xlim(0, max(x[0]))
    ax.set_ylim(0)

    ax.legend(fontsize='x-large')
    ax.tick_params(labelsize='x-large')
    plt.title(r'Histograms with $\mathrm{\Delta z=' + str(dz) + '}$', fontsize='x-large')

    if save:
        if path is not None:
            filename = f'{path}dz={dz:.3f}.png'
        else:
            filename = f'dz={dz:.3f}.png'
        plt.savefig(filename, dpi=150)
    else:
        plt.show()

    plt.close()


def r_hists(dr, w, save=False, path=None):
    from data_processing import distance
    from data_processing import z_final
    from distance import r_comoving

    print('dr=' + str(dr))

    x = []
    y = []
    for wi in w:
        data = distance(w=wi)
        # data = [r_comoving(i) for i in z_med_pdz(w=wi)]
        xx, yy = data_hist(data, dr)
        x.append(xx)
        y.append(yy)

    fig, ax = plt.subplots(figsize=(12.8 * frac, 7.2 * frac))

    if w[0] == -100:
        label = 'all w, N={:d}'.format(sum(y[0]))
        ax.hist(x[0][:-1], bins=x[0], weights=y[0], histtype='step', label=label)
    else:
        label = 'w>{}, N={:d}'.format(str(w[0]), sum(y[0]))
        ax.hist(x[0][:-1], bins=x[0], weights=y[0], histtype='stepfilled', alpha=0.9, label=label)

    for i in range(1, len(w)):
        label = 'w>{}, N={:d}'.format(str(w[i]), sum(y[i]))
        if i == 4:
            ax.hist(x[i][:-1], bins=x[i], weights=y[i], histtype='stepfilled', color='#FFD94A', alpha=0.9, label=label)
        else:
            ax.hist(x[i][:-1], bins=x[i], weights=y[i], histtype='stepfilled', alpha=0.9, label=label)

    ax.set_xlabel('r', fontsize='x-large')
    ax.set_ylabel(r'$\mathrm{\Delta N(R,\Delta R)}$', fontsize='x-large')

    ax.set_xlim(0, max(x[0]))
    ax.set_ylim(0)

    ax.legend(fontsize='x-large')
    ax.tick_params(labelsize='x-large')
    plt.title(r'Histograms with $\mathrm{\Delta R=' + str(dr) + '}$', fontsize='x-large')

    if save:
        if path is not None:
            filename = f'{path}dr={dr}.png'
        else:
            filename = f'dr={dr}.png'
        plt.savefig(filename, dpi=150)
    else:
        plt.show()

    plt.close()


def z_hists(data, dz):
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator

    x, y = [], []
    for k in dz:
        for t in data:
            xx, yy = data_hist(t, k)
            x.append(xx)
            y.append(yy)

    fig = plt.figure(figsize=(15, 8))
    axes = []
    for i in range(8):
        axes.append(fig.add_subplot(4, 2, i + 1))
        axes[i].hist(x[i][:-1], bins=x[i], weights=y[i])
        axes[i].set_xlim(0, 6)
        axes[i].xaxis.set_major_locator(MultipleLocator(1))
        axes[i].xaxis.set_minor_locator(AutoMinorLocator(2))

    axes[0].set_ylabel('N(z)')
    axes[2].set_ylabel('N(z)')
    axes[4].set_ylabel('N(z)')
    axes[6].set_ylabel('N(z)')
    axes[6].set_xlabel('z')
    axes[7].set_xlabel('z')

    axes[0].set_title('Z_MIN_CHI2')
    axes[1].set_title('Z_MED_PDZ')

    fig.tight_layout()
    plt.show()


def ra_dec_plots(data):
    fig = plt.figure(figsize=(12.8, 7.2))

    axes = []
    for i in range(len(data)):
        axes.append(fig.add_subplot((len(data) - 1) // 2 + 1, 2, i + 1))
        axes[i].scatter(x=data[i][0], y=data[i][1], s=0.5)
        axes[i].axis('scaled')

    fig.tight_layout()
    plt.show()


def density_plot_old(data, n, title=None, save=False, path=''):
    import data_processing as dp
    from data_processing import density
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 7.2))

    ax1.scatter(x=data[0], y=data[1], s=0.5)

    den = density(data, n)
    im = ax2.imshow(den, extent=(0, n, n, 0), interpolation='gaussian', cmap='jet')

    ax1.axis('scaled')
    ax1.grid(linestyle='--', alpha=0.3)
    ax1.set_xlabel('RA, [deg]', fontsize='x-large')
    ax1.set_ylabel('Dec, [deg]', fontsize='x-large')
    ax1.tick_params(labelsize='large')

    x_pixel = (dp.max_ra - dp.min_ra) * 60 / n
    y_pixel = (dp.max_dec - dp.min_dec) * 60 / n

    ax2.set_xlim([0, n])
    ax2.set_ylim([0, n])
    ax2.set_xlabel(f'1 pixel = {x_pixel:.1f} arcmin', fontsize='x-large')
    ax2.set_ylabel(f'1 pixel = {y_pixel:.1f} arcmin', fontsize='x-large')
    ax2.tick_params(labelsize='large')

    # divider = make_axes_locatable(axes[2 * i + 1])
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, shrink=0.685, pad=0.01)
    cbar.ax.tick_params(labelsize='large')

    if title is not None:
        plt.suptitle(title, y=0.91, fontsize='xx-large')

    if save:
        filename = f'{path}{title.lower()}.png'
        plt.savefig(filename, dpi=150)
    else:
        plt.show()

    plt.close()


def make_heights_equal(fig, rect, ax1, ax2, pad):
    # pad in inches

    h1, v1 = Size.AxesX(ax1), Size.AxesY(ax1)
    h2, v2 = Size.AxesX(ax2), Size.AxesY(ax2)

    pad_v = Size.Scaled(1)
    pad_h = Size.Fixed(pad)

    my_divider = HBoxDivider(fig, rect,
                             horizontal=[h1, pad_h, h2],
                             vertical=[v1, pad_v, v2])

    ax1.set_axes_locator(my_divider.new_locator(0))
    ax2.set_axes_locator(my_divider.new_locator(2))


def density_plot(ra_dec_data, density_data, max_density, n_ra, n_dec, p, title=None, save=False, path=''):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.ticker import FormatStrFormatter

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8 * frac, 7.2 * frac))

    x = ra_dec_data[0]
    y = ra_dec_data[1]
    ax1.scatter(x=x, y=y, s=0.5)

    im = ax2.imshow(density_data, vmax=max_density, extent=(0, n_ra, n_dec, 0), interpolation='gaussian', cmap='seismic')

    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)
    dx = (max_x - min_x) / 50
    dy = (max_y - min_y) / 50
    ax1.set_xlim(min_x - dx, max_x + dx)
    ax1.set_ylim(min_y - dy, max_y + dy)

    # ax1.axis('scaled')
    ax1.grid(linestyle='--', alpha=0.3)
    ax1.set_xlabel('RA, [deg]', fontsize='x-large')
    ax1.set_ylabel('Dec, [deg]', fontsize='x-large')
    ax1.tick_params(labelsize='large')

    ax2.set_xlim([0, n_ra])
    ax2.set_ylim([0, n_dec])
    # ax2.axis('scaled')
    ax2.set_xlabel(f'1 pixel = {p:.1f} arcmin', fontsize='x-large')
    ax2.set_ylabel(f'1 pixel = {p:.1f} arcmin', fontsize='x-large')
    ax2.tick_params(labelsize='large')

    make_heights_equal(fig, 111, ax1, ax2, pad=0.75)

    axins2 = inset_axes(ax2, width="5%", height="100%", loc='right', borderpad=-2.3 * frac)

    # if the numbers on the x-axis are superimposed on each other
    if frac < 1:
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    cbar = fig.colorbar(im, cax=axins2)
    cbar.ax.tick_params(labelsize='large')

    plt.tight_layout(rect=(0, 0, 0.95, 1))

    if title is not None:
        plt.suptitle(title, y=0.91, fontsize='xx-large')

    if save:
        filename = f'{path}{title.lower()}.png'
        plt.savefig(filename, dpi=150)
    else:
        plt.show()

    plt.close()


def z_vs_z(z1, sigma1, z2, sigma2):
    fig, ax = plt.subplots(figsize=(12.8, 7.2))

    ax.errorbar(z1, z2, yerr=sigma2, fmt='o', elinewidth=0.5, capsize=2)
    x = np.linspace(-1.6, 0.8, 100)
    ax.plot(x, x, c='red')

    # ax.set_xscale('logit')
    # ax.set_yscale('logit')
    ax.set_xlabel('Z_FINAL')
    ax.set_ylabel('Z_MED_PDZ')
    # ax.set_xlim(0, 6)
    # ax.set_ylim(0, 6)
    plt.title('log Z_MED_PDZ vs log Z_FINAL with w>0.7')

    fig.tight_layout()
    plt.show()
    filename = 'med vs final logit'
    # plt.savefig(filename, dpi=150)
    plt.close()


def app_graph_r(data, dr, w, save=False, path=None):
    print(f'dz={dr}, w>{w}')

    x, y = data_hist(data, dr, bi='center')

    # par1, s1 = ap.fast(x, y, 1)
    # par2, s2 = ap.fast(x, y, 2)
    par3, s3 = ap.fast(x, y, 3)

    # a1, b1, c1 = par1
    # a2, b2, z2, c2 = par2
    a3, b3, d3, c3 = par3

    # p = ['{:.2f}'.format(k) for k in [a3, b3]]
    # a3, b3 = p
    # d3 = f'{d3:.0e}'
    # c3 = f'{c3:.0e}'

    x0 = np.linspace(x[0], x[-1], 1000)

    fig, ax = plt.subplots(figsize=(12.8 * frac, 7.2 * frac))
    ax.scatter(x, y, s=10, c=[[1, 0, 1] for i in x])
    ax.plot(x, y)

    # label1 = r'$\mathrm{' + str(int(c1)) + 'z^{' + a1 + '}e^{-' + b1 + 'z}}$'
    # label1 += ', ' + '%.3e' % s1
    # ax.plot(x0, ap.f(x0, par1, 1), color='orange', linestyle='--', linewidth=0.8, label=label1)

    # label2 = r'$\mathrm{' + str(int(c2)) + 'z^{' + a2 + r'}e^{\left(-\frac{z}{' + z2 + r'}\right)^{' + b2 + '}}}$'
    # label2 += ', ' + '%.3e' % s2
    # ax.plot(x0, ap.f(x0, par2, 2), color='limegreen', linestyle='--', linewidth=0.8, label=label2)

    # label3 = r'$\mathrm{' + c3 + r'\left(\frac{R^{' + a3 + '}+R^{' + a3 + r'\cdot' + b3 + '}}{R^{' + b3 + \
    #          '}+' + d3 + r'}\right)}$'
    # label3 += ', ' + '%.3e' % s3
    ax.plot(x0, ap.f(x0, par3, 3), color='orangered', linestyle='--', linewidth=0.8)

    ax.set_xlabel('R', fontsize='x-large')
    ax.set_ylabel(r'$\mathrm{\Delta N(R,\Delta R)}$', fontsize='x-large')

    ax.set_xlim(0)
    ax.set_ylim(0)

    if w == -100:
        plt.title('Approximations with dR={}, all w, N={:d}'.format(str(dr), sum(y)),
                  fontsize='x-large')
    else:
        plt.title('Approximations with dR={}, w>{}, N={:d}'.format(str(dr), str(w), sum(y)),
                  fontsize='x-large')

    # ax.legend(fontsize='x-large')

    # if the legend is superimposed on the graph
    # ax.legend(fontsize='x-large', loc='lower left', bbox_to_anchor=(0.05, 0))

    ax.tick_params(labelsize='x-large')

    if save:
        if path is not None:
            filename = path + 'dR={:.3f}.png'.format(dr)
        else:
            filename = 'dR={:.3f}.png'.format(dr)
        plt.savefig(filename, dpi=150)
    else:
        plt.show()

    plt.close()
