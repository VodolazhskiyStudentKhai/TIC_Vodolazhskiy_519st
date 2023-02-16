import numpy
import scipy
import matplotlib.pyplot as plt

'''
Лабораторна робота №2
студента групи 519ст
Водолажського Владислава
Варіант 6
'''


def build_graphic():
    n = 500
    fs = 1000
    fmax = 13

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
    ...


build_graphic()
