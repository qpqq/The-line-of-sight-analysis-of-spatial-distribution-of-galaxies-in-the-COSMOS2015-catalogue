import numpy as np
from math import ceil
from path import output_name
from distance import r_comoving


min_ra, max_ra, min_dec, max_dec = 149.4114, 150.7866, 1.614532, 2.814121


def indexes():
    file = open(output_name, 'r')

    for i in range(2):
        file.readline()

    line = file.readline()
    for i, j in enumerate(line):
        print(i, j)


def data_proc(ind=False):
    from path import input_name

    file = open(input_name, 'r')
    output = open(output_name, 'w')

    if ind:
        k = 147
        for i in range(k):
            if i % 100 == 0:
                output.write(str(i // 100 % 100))
            else:
                output.write(' ')
        output.write('\n')

        for i in range(k):
            if i % 10 == 0:
                output.write(str(i // 10 % 10))
            else:
                output.write(' ')
        output.write('\n')

        for i in range(k):
            output.write(str(i % 10))
        output.write('\n')

    line = file.readline()
    output.write(line[:43] + line[56:-1] + '     W_MED_PDZ    R_COMOVING\n')
    output.write('\n')

    for line in file:
        if line[55] == '0':
            output.write(line[:43] + line[56:-1])

            z_pdz = float(line[72:81]) * 10 ** int(line[82:86])
            z_fin = float(line[57:66]) * 10 ** int(line[67:71])

            if 0 < z_pdz < 6:

                z_min68 = float(line[87:96]) * 10 ** int(line[97:101])
                z_max68 = float(line[102:111]) * 10 ** int(line[112:116])
                dz = (z_max68 - z_min68) / 2

                sigma = (z_pdz - dz) / z_pdz
                if 0 < z_fin < 6:
                    dist = float(r_comoving(z_fin))
                else:
                    dist = -9.999999 * 10 ** 9

                output.write(' % E % E\n' % (sigma, dist))

            else:
                output.write(' -9.999999E+09 -9.999999E+09\n')

    file.close()
    output.close()


def z_final(min_z=0, max_z=6, w=-100, err='None', log=False, file=None):
    from math import log10
    from astropy.table import Table

    if file:
        file = Table.read(f'{file}.csv', format='ascii.csv')

        data = []
        for i in range(len(file)):
            z = file['Z_FINAL'][i]

            if min_z < z < max_z and file['W_MED_PDZ'][i] > w:
                data.append(z)

        return sorted(data)

    file = open(output_name, 'r')

    for i in range(2):
        file.readline()

    data = []
    for line in file:
        z = float(line[44:53]) * 10 ** int(line[54:58])

        if min_z < z < max_z and float(line[119:128]) * 10 ** int(line[129:132]) > w:
            if log:
                z = log10(z)
            if err == 'None':
                data.append(z)
            elif err == 'abs':
                z_min = float(line[74:83]) * 10 ** int(line[84:88])
                z_max = float(line[89:98]) * 10 ** int(line[99:103])
                data.append([z, z_min / z, z_max / z])
            elif err == 'rel':
                z_min = float(line[74:83]) * 10 ** int(line[84:88])
                z_max = float(line[89:98]) * 10 ** int(line[99:103])
                data.append([z, z_min, z_max])

    file.close()
    return sorted(data)


def z_med_pdz(min_z=0, max_z=6, w=-100, err='None', log=False):
    from math import log10

    file = open(output_name, 'r')

    for i in range(2):
        file.readline()

    data = []
    for line in file:
        z = float(line[59:68]) * 10 ** int(line[69:73])

        if min_z < z < max_z and float(line[119:128]) * 10 ** int(line[129:132]) > w:
            if log:
                z = log10(z)
            if err == 'None':
                data.append(z)
            elif err == 'abs':
                z_min = float(line[74:83]) * 10 ** int(line[84:88])
                z_max = float(line[89:98]) * 10 ** int(line[99:103])
                data.append([z, z_min / z, z_max / z])
            elif err == 'rel':
                z_min = float(line[74:83]) * 10 ** int(line[84:88])
                z_max = float(line[89:98]) * 10 ** int(line[99:103])
                data.append([z, z_min, z_max])

    file.close()
    return sorted(data)


def z_min_chi2(min_z=0, max_z=6, w=-100, err='None', log=False):
    from math import log10

    file = open(output_name, 'r')

    for i in range(2):
        file.readline()

    data = []
    for line in file:
        z = float(line[105:113]) * 10 ** int(line[114:118])

        if min_z < z < max_z and float(line[119:128]) * 10 ** int(line[129:132]) > w:
            if log:
                z = log10(z)
            if err == 'None':
                data.append(z)
            elif err == 'abs':
                z_min = float(line[74:83]) * 10 ** int(line[84:88])
                z_max = float(line[89:98]) * 10 ** int(line[99:103])
                data.append([z, z_min / z, z_max / z])
            elif err == 'rel':
                z_min = float(line[74:83]) * 10 ** int(line[84:88])
                z_max = float(line[89:98]) * 10 ** int(line[99:103])
                data.append([z, z_min, z_max])

    file.close()
    return sorted(data)


def distance(min_d=0, max_d=-1, w=-100):
    if max_d == -1:
        max_d = r_comoving(6)

    file = open(output_name, 'r')

    for i in range(2):
        file.readline()

    data = []
    for line in file:
        d = float(line[134:142]) * 10 ** int(line[143:146])

        if min_d < d < max_d and float(line[119:128]) * 10 ** int(line[129:132]) > w:
            data.append(d)

    file.close()
    return sorted(data)


def ra_dec(column, left=-10 ** 10, right=10 ** 10, w=-100):
    file = open(output_name, 'r')
    column = column.upper()

    for i in range(2):
        file.readline()

    ra = []
    dec = []

    for line in file:
        if column == 'Z_FINAL':
            if left < float(line[44:53]) * 10 ** int(line[54:58]) < right \
                    and float(line[119:128]) * 10 ** int(line[129:132]) > w:
                dec.append(float(line[29:38]) * 10 ** int(line[39:43]))
                ra.append(float(line[14:23]) * 10 ** int(line[24:28]))

        elif column == 'Z_MED_PDZ':
            if left < float(line[59:68]) * 10 ** int(line[69:73]) < right \
                    and float(line[119:128]) * 10 ** int(line[129:132]) > w:
                dec.append(float(line[29:38]) * 10 ** int(line[39:43]))
                ra.append(float(line[14:23]) * 10 ** int(line[24:28]))

        elif column == 'DIST_MED_PDZ':
            if left < float(line[133:142]) * 10 ** int(line[143:146]) < right \
                    and float(line[119:128]) * 10 ** int(line[129:132]) > w:
                dec.append(float(line[29:38]) * 10 ** int(line[39:43]))
                ra.append(float(line[14:23]) * 10 ** int(line[24:28]))

    file.close()
    return ra, dec


def data_hist(data, delta, bi='edges'):
    n = 1
    while delta * n <= data[-1]:
        n += 1
    bins = [i * delta for i in range(n + 1)]

    j = 0
    counts = []
    first = False
    k = 0
    for i in range(1, len(bins)):
        q = 0

        while j < len(data) and data[j] < bins[i]:
            first = True
            q += 1
            j += 1

        if q == 0 and not first:
            k += 1
            continue
        counts.append(q)

    bins = bins[k:]
    if bi == 'edges':
        return bins, counts
    elif bi == 'center':
        return [bins[i] + delta / 2 for i in range(len(bins) - 1)], counts


def density(data, n):
    minx = min(data[0])
    miny = min(data[1])
    dx = (max(data[0]) - minx) / n
    dy = (max(data[1]) - miny) / n

    number = len(data[0])

    q = [[0 for i in range(n)] for j in range(n)]

    for i in range(number):
        x = data[0][i] - minx
        y = data[1][i] - miny
        i = int(x // dx)
        j = int(y // dy)

        if i != n and j != n:
            q[j][i] += 1
        elif i == n and j != n:
            q[j][-1] += 1
        elif i != n and j == n:
            q[-1][i] += 1
        else:
            q[-1][-1] += 1

    return q


def density_plot_data(start, finish, delta, step, p, w):
    st = np.around(float(start), 3)
    fin = np.around(start + delta, 3)

    n_ra = ceil((max_ra - min_ra) * 60 / p)
    n_dec = ceil((max_dec - min_dec) * 60 / p)

    data = []
    st_list = []
    fin_list = []

    i = 0
    while fin <= finish:
        data.append([[[], []], [[0 for j in range(n_ra)] for k in range(n_dec)]])
        st_list.append(st)
        fin_list.append(fin)

        st = np.around(st + step, 3)
        fin = np.around(fin + step, 3)

        i += 1

    file = open(output_name, 'r')
    for i in range(2):
        file.readline()

    for line in file:
        z = float(line[44:53]) * 10 ** int(line[54:58])

        if start < z < finish and float(line[119:128]) * 10 ** int(line[129:132]) > w:
            st_set = set()
            fin_set = set()

            i = 0
            while i < len(st_list) and st_list[i] <= z:
                st_set.add(i)
                i += 1

            i = len(fin_list) - 1
            while i > -1 and fin_list[i] >= z:
                fin_set.add(i)
                i -= 1

            ra = float(line[14:23]) * 10 ** int(line[24:28])
            dec = float(line[29:38]) * 10 ** int(line[39:43])

            ra_ind = int(((ra - min_ra) * 60) // p)
            dec_ind = int(((dec - min_dec) * 60) // p)
            if ra_ind == n_ra:
                ra_ind = -1
            if dec_ind == n_dec:
                dec_ind = -1

            for i in st_set.intersection(fin_set):
                data[i][0][0].append(ra)
                data[i][0][1].append(dec)

                data[i][1][dec_ind][ra_ind] += 1

    file.close()
    return data, n_ra, n_dec


def data_log(data, n, alpha, bi='edges'):
    bins = [max(data) * (i / n) ** alpha for i in range(n + 1)]
    y = [0 for i in range(n)]

    m = 0
    for i in range(n):
        for j in range(m, len(data)):
            if data[j] < bins[i + 1]:
                y[i] += 1
            else:
                m = j
                break

    if bi == 'edges':
        return bins, y
    elif bi == 'center':
        return [(bins[i - 1] + bins[i]) / 2 for i in range(1, len(bins))], y
