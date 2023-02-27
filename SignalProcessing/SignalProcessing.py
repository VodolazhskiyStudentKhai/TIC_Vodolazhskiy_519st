import numpy
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

    e1 = discrete_signal_after_filers - filtered_signal
    print(e1)
    print("______________________")

    x1 = [2, 4, 8, 16]
    e2 = e1[0][0:len(x1)]
    signal_noise = numpy.var(e1)/numpy.var(filtered_signal)
    signal_noise2 = []
    for i in range(len(x1)):
        signal_noise2.append(signal_noise - e1[0][i])

    print(x1)
    print(signal_noise)
    x_lab = "Час (секунд)"
    y_lab = "Амплітуда сигналу"
    title_disc = "Сигнал з кроком дискретизації Dt = (2, 4, 8, 16)"

    draw2(time_line_ox, discrete_signals, x_lab, y_lab, title_disc, "графік 3")
    draw2(freq_countdown, discrete_spectrums, x_lab, "Амплітуда спектру", title_disc, "графік 4")
    draw2(time_line_ox, discrete_signal_after_filers, x_lab, y_lab, title_disc, "графік 5")
    draw1(x1, e2, "Крок дискретизації", "Дисперсія", "Залежність дисперсії від кроку дискретизації", "графік 6")
    draw1(x1, signal_noise2, "Крок дискретизації", "ССШ", "Залежність дисперсії від кроку дискретизації", "графік 7")

    ...


def draw2(x, y, x_lab: str, y_lab: str, title: str, file_name: str):
    s = 0
    fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
    for i in range(0, 2):
        for j in range(0, 2):
            ax[i][j].plot(x, y[s], linewidth=1)
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
