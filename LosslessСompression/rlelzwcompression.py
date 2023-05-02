import ast
import collections
import math
from matplotlib import pyplot as plt

open("results_rle_lzw.txt", "w", encoding="utf-8")
result_list = []
n_seq = 100


def encode_rle(sequence):
    res = []
    el = 1
    for i, j in enumerate(sequence):
        if i == 0:
            continue
        elif j == sequence[i - 1]:
            el += 1
        else:
            res.append((sequence[i - 1], el))
            el = 1
    res.append((sequence[len(sequence) - 1], el))

    return_list = []
    for i, item in enumerate(res):
        return_list.append(f"{item[1]}{item[0]}")

    return "".join(return_list), res


def encode_lzw(sequence):
    dt = {}
    for i in range(65536):
        dt[chr(i)] = i
    el = ""
    res = []
    max_size = 0
    for char in sequence:
        n = el + char
        if n in dt:
            el = n
        else:
            res.append(dt[el])
            dt[n] = len(dt)
            b = 16 if dt[el] < 65536 else math.ceil(math.log2(len(dt)))
            el = char
            with open("results_rle_lzw.txt", "a", encoding="utf-8") as f:
                f.write(f"Code: {dt[el]}, Element: {el}, Bits: {b}\n")
                f.close()
            max_size = max_size + b
    last = 16 if dt[el] < 65536 else math.ceil(math.log2(len(dt)))
    max_size = max_size + last
    with open("results_rle_lzw.txt", "a", encoding="utf-8") as f:
        f.write(f"Code: {dt[el]}, Element: {el}, Bits: {last}\n")
    res.append(dt[el])
    return res, max_size


def decode_rle(sequence):
    res = []
    for item in sequence:
        res.append(item[0] * item[1])
    return "".join(res)


def decode_lzw(sequence):
    dictionary = {}
    for i in range(65536):
        dictionary[i] = chr(i)
    res = ""
    last = None

    for code in sequence:
        if code in dictionary:
            curr = dictionary[code]
            res += curr
            if last is not None:
                dictionary[len(dictionary)] = last + curr[0]
            last = curr
        else:
            curr = last + last[0]
            res += curr
            dictionary[len(dictionary)] = curr
            last = curr
    return res


def start():
    with open("sequence.txt", "r", encoding="utf-8") as file:
        first_seq = ast.literal_eval(file.read())
        first_seq = [i.strip("[]',") for i in first_seq]
        file.close()

    for seq in first_seq:
        counter = collections.Counter(seq)
        prob = {symbol: count / n_seq for symbol, count in counter.items()}
        entropy = -sum(p * math.log2(p) for p in prob.values())

        file = open("results_rle_lzw.txt", "a", encoding="utf-8")
        file.write('Оригінальна послідовність: {0}\n'.format(str(seq)))
        file.write('Розмір оригінальної послідовності: {0} bits\n'.format(str(len(seq) * 16)))
        file.write('Ентропія: {0}\n'.format(round(entropy, 2)))
        file.write('\n')
        file.close()

        encoded_seq, encoded = encode_rle(seq)
        decoded_seq = decode_rle(encoded)
        total_rle = len(encoded_seq) * 16
        comp_ratio_rle = round((len(seq) / len(encoded_seq)), 3)

        if comp_ratio_rle < 1:
            comp_ratio_rle = '-'

        file = open("results_rle_lzw.txt", "a", encoding="utf-8")
        file.write(
            '_________________________________________Кодування_RLE_______________________________________' + '\n')
        file.write('Закодована RLE послідовність: ' + str(encoded_seq) + '\n')
        file.write('Розмір закодованої RLE послідовності: ' + str(total_rle) + ' bits' + '\n')
        file.write('Коефіцієнт стиснення RLE: ' + str(comp_ratio_rle) + '\n')
        file.close()

        file = open("results_rle_lzw.txt", "a", encoding="utf-8")
        file.write('Декодована RLE послідовність: ' + str(decoded_seq) + '\n')
        file.write('Розмір декодованої RLE послідовності: ' + str(len(decoded_seq) * 16) + ' bits' + '\n')
        file.close()

        with open("results_rle_lzw.txt", "a", encoding="utf-8") as file:
            file.write(
                '_________________________________________Кодування_LZW_________________________________________'
                + '\n')
            file.write(
                '_________________________________________Поетапне кодування_________________________________________'
                + '\n')

        result_encoded, size = encode_lzw(seq)
        with open("results_rle_lzw.txt", "a", encoding="utf-8") as file:
            file.write('________________________________________________________________________________' + '\n')
            file.write(f"Закодована LZW послідовність:{''.join(map(str, result_encoded))} \n")
            file.write(f"Розмір закодованої LZW послідовності: {size} bits \n")
            compression_ratio_lzw = round((len(seq) * 16 / size), 3)

            if compression_ratio_lzw < 1:
                compression_ratio_lzw = '-'
            else:
                compression_ratio_lzw = compression_ratio_lzw

            file.write(f"Коефіціент стиснення LZW: {compression_ratio_lzw} \n")
            file.close()

        decoded_result_lzw = decode_lzw(result_encoded)
        with open("results_rle_lzw.txt", "a", encoding="utf-8") as file:
            file.write(f"Декодована LZW послідовність:{''.join(map(str, decoded_result_lzw))} \n")
            file.write(f"Розмір декодованої LZW послідовності: {len(decoded_result_lzw) * 16} bits \n ")
            file.write('\n' + '\n' + '\n' + '\n' + '\n')
            file.close()

        result_list.append([round(entropy, 3), comp_ratio_rle, compression_ratio_lzw])

    fig, ax = plt.subplots(figsize=(14 / 1.54, 8 / 1.54))
    headers = ['Ентропія', 'КС RLE', 'КС LZW']
    row = ['Послідовність 1', 'Послідовність 2', 'Послідовність 3', 'Послідовність 4', 'Послідовність 5',
           'Послідовність 6', 'Послідовність 7', 'Послідовність 8']
    ax.axis('off')
    table = ax.table(cellText=result_list, colLabels=headers, rowLabels=row,
                     loc='center', cellLoc='center')

    table.set_fontsize(14)
    table.scale(0.8, 2)
    ax.text(0.5, 0.95, 'Результати стиснення методами RLE та LZW',
            transform=ax.transAxes, ha='center', va='top', fontsize=14)

    fig.savefig('Результати стиснення методами RLE та LZW' + '.jpg', dpi=600)


start()
