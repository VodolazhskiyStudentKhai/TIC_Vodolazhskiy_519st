import random
import string
import collections
import math

from matplotlib import pyplot as plt


def lossless_comp():

    def parameters(s):
        counter = collections.Counter(s)
        probab = {symbol: count / n_seq for symbol, count in counter.items()}
        probability_string = ", ".join(
            [f"{symbol}={prob:.2f}" for symbol, prob in probab.items()]
        )
        main_probability = sum(probab.values()) / len(probab)
        equality = all(
            abs(prob - main_probability) < 0.05 * main_probability
            for prob in probab.values()
        )
        u = "рівна" if equality == main_probability else "нерівна"

        entrop = -sum(p * math.log2(p) for p in probab.values())
        if sequence_size_alphabet > 1:
            s_ex = 1 - entrop / math.log2(sequence_size_alphabet)
        else:
            s_ex = 1
        return probability_string, main_probability, u, entrop, s_ex

    text = open("results_sequence.txt", "w")
    # №1
    n_seq = 100
    n1 = 6  # варіант
    arr1 = [1] * n1
    n0 = n_seq - n1
    arr0 = [0] * n0
    results = []
    os = arr1 + arr0
    random.shuffle(os)
    os = "".join(map(str, os))
    text.write(f"Варіант 6\n Водолажський 519ст\n")
    text.write(f"Завдання 1\n")
    text.write("Послідовність: " + str(os) + "\n")
    original_sequence_size = len(os)
    text.write("Розмір послідовності: " + str(original_sequence_size) + " byte" + "\n")
    uniq_ch = set(os)
    sequence_size_alphabet = len(uniq_ch)
    text.write("Розмір алфавіту: " + str(sequence_size_alphabet) + "\n")
    prob_str, mean_prob, uniform, entropy, source_exc = parameters(
        os
    )
    results.append(
        [sequence_size_alphabet, round(entropy, 2), round(source_exc, 2), uniform]
    )
    write(text, prob_str, mean_prob, uniform, entropy, source_exc)

    # №2
    list2 = ["В", "о", "д", "о", "л", "а", "ж", "с", "ь", "к", "и", "й"]
    n0_2 = n_seq - len(list2)
    list0_2 = [0] * n0_2
    os_2 = list2 + list0_2
    os_2 = "".join(map(str, os_2))
    text.write(f"Завдання 2\n")
    text.write("Послідовність: " + str(os_2) + "\n")
    text.write(
        "Розмір послідовності: " + str(len(os)) + " byte" + "\n"
    )
    sequence_size_alphabet = len(uniq_ch)
    text.write("Розмір алфавіту: " + str(sequence_size_alphabet) + "\n")

    prob_str, mean_prob, uniform, entropy, source_exc = parameters(
        os_2
    )
    results.append(
        [sequence_size_alphabet, round(entropy, 2), round(source_exc, 2), uniform]
    )

    write(text, prob_str, mean_prob, uniform, entropy, source_exc)

    # №3
    os_3 = list(os_2)
    random.shuffle(os_3)
    os_3 = "".join(map(str, os_3))
    text.write(f"Завдання 3\n")
    text.write("Послідовність: " + str(os_3) + "\n")
    text.write(
        "Розмір послідовності: " + str(len(os_3)) + " byte" + "\n"
    )
    uniq_ch = set(os_3)
    sequence_size_alphabet = len(uniq_ch)
    text.write("Розмір алфавіту: " + str(sequence_size_alphabet) + "\n")

    prob_str, mean_prob, uniform, entropy, source_exc = parameters(
        os_3
    )
    results.append(
        [sequence_size_alphabet, round(entropy, 2), round(source_exc, 2), uniform]
    )

    write(text, prob_str, mean_prob, uniform, entropy, source_exc)

    # №4
    aray = []
    letters = ["В", "о", "д", "о", "л", "а", "ж", "с", "ь", "к", "и", "й", "5", "1", "9", "c", "т"]
    n_letters = len(letters)
    n_repeats = n_seq / n_letters
    remainder = n_seq * (n_seq % n_letters)
    aray += letters * int(n_repeats)
    aray += letters[:remainder]
    os_4 = "".join(map(str, aray))
    text.write(f"Завдання 4\n")
    text.write("Послідовність: " + os_4 + "\n")
    text.write(
        "Розмір послідовності: " + str(len(os_4)) + " byte" + "\n"
    )
    uniq_ch = set(os_4)
    sequence_size_alphabet = len(uniq_ch)
    text.write("Розмір алфавіту: " + str(sequence_size_alphabet) + "\n")

    prob_str, mean_prob, uniform, entropy, source_exc = parameters(
        os_4
    )
    results.append(
        [sequence_size_alphabet, round(entropy, 2), round(source_exc, 2), uniform]
    )

    write(text, prob_str, mean_prob, uniform, entropy, source_exc)

    # №5
    alphabet = ["в", "о", "5", "1", "9"]
    p_i = 0.2
    length = p_i * n_seq
    os_5 = alphabet * int(length)
    random.shuffle(os_5)
    os_5 = "".join(map(str, os_5))
    text.write(f"Завдання 5\n")
    text.write("Послідовність: " + str(os_5) + "\n")
    text.write(
        "Розмір послідовності: " + str(len(os_5)) + " byte" + "\n"
    )
    uniq_ch = set(os_5)
    sequence_size_alphabet = len(uniq_ch)
    text.write("Розмір алфавіту: " + str(sequence_size_alphabet) + "\n")

    counts = collections.Counter(os_5)
    probability = {symbol: count / n_seq for symbol, count in counts.items()}
    prob_str = ", ".join(
        [f"{symbol}={prob:.4f}" for symbol, prob in probability.items()]
    )
    mean_prob = sum(probability.values()) / len(probability)
    equal = all(
        abs(prob - mean_prob) < 0.05 * mean_prob
        for prob in probability.values()
    )
    uniform = "рівна" if equal else "нерівна"

    entropy = -sum(p * math.log2(p) for p in probability.values())
    if sequence_size_alphabet > 1:
        source_exc = 1 - entropy / math.log2(sequence_size_alphabet)
    else:
        source_exc = 1
    results.append(
        [sequence_size_alphabet, round(entropy, 2), round(source_exc, 2), uniform]
    )

    write(text, prob_str, mean_prob, uniform, entropy, source_exc)

    # №6
    letter_list = ["в", "о"]
    digit_list = ["5", "1", "9"]
    p_let = 0.7
    p_dig = 0.3
    n_letters6 = int(p_let * n_seq) / len(letter_list)
    n_digits6 = int(p_dig * n_seq) / len(digit_list)
    l_list = letter_list * int(n_letters6)
    d_list = digit_list * int(n_digits6)
    os_6 = l_list + d_list
    random.shuffle(os_6)
    os_6 = "".join(map(str, os_6))
    text.write(f"Завдання 6\n")
    text.write("Послідовність: " + str(os_6) + "\n")
    text.write(
        "Розмір послідовності: " + str(len(os_6)) + " byte" + "\n"
    )
    uniq_ch = set(os_6)
    sequence_size_alphabet = len(uniq_ch)
    text.write("Розмір алфавіту: " + str(sequence_size_alphabet) + "\n")

    prob_str, mean_prob, uniform, entropy, source_exc = parameters(
        os_6
    )
    results.append(
        [sequence_size_alphabet, round(entropy, 2), round(source_exc, 2), uniform]
    )

    write(text, prob_str, mean_prob, uniform, entropy, source_exc)

    # №7
    elements = string.ascii_lowercase + string.digits
    os_7 = [random.choice(elements) for _ in range(n_seq)]
    os_7 = "".join(map(str, os_7))
    text.write("Послідовність: " + str(os_7) + "\n")
    text.write(
        "Розмір послідовності: " + str(len(os_7)) + " byte" + "\n"
    )
    uniq_ch = set(os_7)
    sequence_size_alphabet = len(uniq_ch)
    text.write(f"Завдання 7\n")
    text.write("Розмір алфавіту: " + str(sequence_size_alphabet) + "\n")

    prob_str, mean_prob, uniform, entropy, source_exc = parameters(
        os_7
    )
    results.append(
        [sequence_size_alphabet, round(entropy, 2), round(source_exc, 2), uniform]
    )

    write(text, prob_str, mean_prob, uniform, entropy, source_exc)

    # №8
    os_8 = ["1"] * n_seq
    os_8 = "".join(map(str, os_8))
    text.write(f"Завдання 8\n")
    text.write("Послідовність: " + str(os_8) + "\n")
    text.write(
        "Розмір послідовності: " + str(len(os_8)) + " byte" + "\n"
    )
    uniq_ch = set(os_8)
    sequence_size_alphabet = len(uniq_ch)
    text.write("Розмір алфавіту: " + str(sequence_size_alphabet) + "\n")

    prob_str, mean_prob, uniform, entropy, source_exc = parameters(
        os_8
    )
    results.append(
        [sequence_size_alphabet, round(entropy, 2), round(source_exc, 2), uniform]
    )

    write(text, prob_str, mean_prob, uniform, entropy, source_exc)

    text.close()

    seq = open("sequence.txt", "w")
    os_list = [
        os,
        os_2,
        os_3,
        os_4,
        os_5,
        os_6,
        os_7,
        os_8,
    ]
    seq.write(str(os_list))
    seq.close()

    text.close()

    fig, ax = plt.subplots(figsize=(14 / 1.54, 8 / 1.54))
    headers = ["Розмір алфавіту", "Ентропія", "Надмірність", "Ймовірність"]
    rows = [
        "Послідовність 1",
        "Послідовність 2",
        "Послідовність 3",
        "Послідовність 4",
        "Послідовність 5",
        "Послідовність 6",
        "Послідовність 7",
        "Послідовність 8",
    ]
    ax.axis("off")
    table = ax.table(
        cellText=results,
        colLabels=headers,
        rowLabels=rows,
        loc="center",
        cellLoc="center",
    )
    table.set_fontsize(14)
    table.scale(0.8, 2)
    fig.savefig("Таблиця" + ".png")


def write(text, prob_str, mean_prob, uniform, entropy, source_ex):
    text.write(f"Ймовірність появи символів: {prob_str}\n")
    text.write(f"Середнє арифметичне ймовірності: {round(mean_prob, 2)}\n")
    text.write(f"Ймовірність розподілу символів: {uniform}\n")
    text.write(f"Ентропія: {round(entropy, 2)}\n")
    text.write(f"Надмірність джерела: {round(source_ex, 2)}\n")
    text.write("\n")


lossless_comp()
