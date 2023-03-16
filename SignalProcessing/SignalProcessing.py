import numpy
import numpy as np
import scipy
import matplotlib.pyplot as plt

'''
Лабораторна робота №2/3
студента групи 519ст
Водолажського Владислава
Варіант 6
'''


def build_graphic():
    n = 500
    fs = 1000
    fmax = 13
    f_fil = 20

    title1 = "Сигнал з максимальною частотою fmax = {0} dpi = 600".format(fmax)
    title2 = "Спектр з максимальною частотою fmax = {0} dpi = 600".format(fmax)
    x_label = "Час(сек)"
    y_label = "Амплітуда"

    # Розрахунок 1
    random = numpy.random.normal(0, 10, n)
    time_line_ox = numpy.arange(n) / fs
    w = fmax / (fs / 2)
    parameters_filter = scipy.signal.butter(3, w, 'low', output='sos')
    filtered_signal = scipy.signal.sosfiltfilt(parameters_filter, random)

    # График 1
    fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
    ax.plot(time_line_ox, filtered_signal, linewidth=1)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    plt.title(title1, fontsize=14)
    ax.grid()
    fig.savefig('./figures/' + 'графік 1' + '.png', dpi=600)

    # Розрахунок 2
    spectrum = scipy.fft.fft(filtered_signal)
    spectrum = numpy.abs(scipy.fft.fftshift(spectrum))
    length_signal = n
    freq_countdown = scipy.fft.fftfreq(length_signal, 1 / length_signal)
    freq_countdown = scipy.fft.fftshift(freq_countdown)

    # График 2
    fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
    ax.plot(freq_countdown, spectrum, linewidth=1)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    plt.title(title2, fontsize=14)
    ax.grid()
    fig.savefig('./figures/' + 'графік 2' + '.png', dpi=600)

    # Практична робота №3
    discrete_signals = []
    discrete_signal_after_filers = []

    w = f_fil / (fs / 2)
    parameters_fil = scipy.signal.butter(3, w, 'low', output='sos')
    filtered_signal2 = None

    for Dt in [2, 4, 8, 16]:
        discrete_signal = numpy.zeros(n)
        for i in range(0, round(n / Dt)):
            discrete_signal[i * Dt] = filtered_signal[i * Dt]
            filtered_signal2 = scipy.signal.sosfiltfilt(parameters_fil, discrete_signal)
            ...
        discrete_signals += [list(discrete_signal)]
        discrete_spectrums = scipy.fft.fft(discrete_signals)
        discrete_spectrums = numpy.abs(scipy.fft.fftshift(discrete_spectrums))
        discrete_signal_after_filers += [list(filtered_signal2)]
        ...

    x1 = [2, 4, 8, 16]
    x_lab = "Час (секунд)"
    y_lab = "Амплітуда сигналу"
    title_disc = "Сигнал з кроком дискретизації Dt = (2, 4, 8, 16)"

    draw2(time_line_ox, discrete_signals, x_lab, y_lab, title_disc, "графік 3")
    draw2(freq_countdown, discrete_spectrums, x_lab, "Амплітуда спектру", title_disc, "графік 4")
    draw2(time_line_ox, discrete_signal_after_filers, x_lab, y_lab, title_disc, "графік 5")

    dispersions = []
    signal_noise = []
    for i in range(len(x1)):
        e1 = discrete_signal_after_filers[i] - filtered_signal
        dispersion = numpy.var(e1)
        dispersions.append(dispersion)
        signal_noise.append(numpy.var(filtered_signal) / dispersion)
        ...

    draw1(x1, dispersions, "Крок дискретизації", "Дисперсія", "Залежність дисперсії від кроку дискретизації",
          "графік 6")
    draw1(x1, signal_noise, "Крок дискретизації", "ССШ", "Залежність відношення сигнал-шум від кроку дискретизації",
          "графік 7")
# Практична робота №4
    bits_list = []
    quantize_signals = []
    num = 0
    x2 = [4, 16, 64, 256]
    for m in x2:
        delta = (numpy.max(filtered_signal) - numpy.min(filtered_signal)) / (m - 1)
        quantize_signal = delta * np.round(filtered_signal / delta)
        quantize_signals.append(list(quantize_signal))
        quantize_levels = numpy.arange(numpy.min(quantize_signal), numpy.max(quantize_signal) + 1, delta)
        quantize_bit = numpy.arange(0, m)
        quantize_bit = [format(bits, '0' + str(int(numpy.log(m) / numpy.log(2))) + 'b') for bits in quantize_bit]
        quantize_table = numpy.c_[quantize_levels[:m], quantize_bit[:m]]
        fig, ax = plt.subplots(figsize=(14 / 2.54, m / 2.54))
        table = ax.table(cellText=quantize_table, colLabels=['Значення сигналу', 'Кодова послідовність'],
                         loc='center')
        table.set_fontsize(14)
        table.scale(1, 2)
        ax.axis('off')
        fig.savefig('./figures/' + 'Таблиця квантування для %d рівнів ' % m + '.png', dpi=600)
        bits = []
        for signal_value in quantize_signal:
            for index, value in enumerate(quantize_levels[:m]):
                if numpy.round(numpy.abs(signal_value - value), 0) == 0:
                    bits.append(quantize_bit[index])
                    break

        bits = [int(item) for item in list(''.join(bits))]
        bits_list.append(bits)
        fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
        ax.step(numpy.arange(0, len(bits)), bits, linewidth=0.1)
        ax.set_xlabel("Біти", fontsize=14)
        ax.set_ylabel("Амплітуда сигналу", fontsize=14)
        plt.title(f'Кодова послідовність при кількості рівнів квантування {m}', fontsize=14)
        ax.grid()
        fig.savefig('./figures/' + 'графік %d ' % (8 + num) + '.png', dpi=600)
        num += 1
        ...
    dispersions = []
    signal_noise = []
    for i in range(len(x2)):
        e1 = quantize_signals[i] - filtered_signal
        dispersion = numpy.var(e1)
        dispersions.append(dispersion)
        signal_noise.append(numpy.var(filtered_signal) / dispersion)
        ...

    draw2(time_line_ox, quantize_signals, "Час(сек)", "Амплітуда сигналу",
          "Цифрові сигнали з рівнями квантування (4, 16, 64, 256)", "графік 12")
    draw1(x2, dispersions, "Кількість рівнів квантування", "Дисперсія",
          "Залежність дисперсії від кількості рівнів квантування", "графік 13")
    draw1(x2, signal_noise, "Кількість рівнів квантування", "ССШ",
          "Залежність відношення сигнал-шум від кількості рівнів квантування", "графік 14")
    ...


def draw2(x, y, x_lab: str, y_lab: str, title: str, file_name: str):
    s = 0
    fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
    for i in range(0, 2):
        for j in range(0, 2):
            ax[i][j].plot(x, y[s], linewidth=1)
            ax[i][j].grid()
            s += 1
            ...
        ...
    fig.supxlabel(x_lab, fontsize=14)
    fig.supylabel(y_lab, fontsize=14)
    fig.suptitle(title, fontsize=14)
    fig.savefig('./figures/' + file_name + '.png', dpi=600)


def draw1(x, y, x_lab: str, y_lab: str, title: str, file_name: str):
    fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))

    ax.plot(x, y, linewidth=2)
    ax.set_xlabel(x_lab, fontsize=14)
    ax.set_ylabel(y_lab, fontsize=14)

    plt.title(title, fontsize=14)
    ax.grid()
    fig.savefig('./figures/' + file_name + '.png', dpi=600)


build_graphic()
