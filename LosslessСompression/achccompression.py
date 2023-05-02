import ast
import collections
import math
import matplotlib.pyplot as plt


def create_file(filename: str):
    with open(filename, 'w', encoding="utf-8") as fl:
        fl.close()


def append_file(file_str: str, message: str):
    with open(file_str, "a", encoding="utf-8") as fl:
        fl.write(message + '\n')
        fl.close()


def enc_ac(uniq_ch, prob, alphabet_len, seq):
    alphabet_list = list(uniq_ch)
    prob = [prob[symbol] for symbol in alphabet_list]
    unity = []

    prob_dict = {alphabet_list[i]: prob[i] for i in range(alphabet_len)}

    prob_range = 0.0
    for i in range(alphabet_len):
        prev = prob_range
        prob_range = prob_range + prob[i]
        unity.append([alphabet_list[i], prev, prob_range])
    for s in seq[:-1]:
        for j in range(len(unity)):
            if s == unity[j][0]:
                probability_low = unity[j][1]
                probability_high = unity[j][2]
                diff = probability_high - probability_low
                for k in range(len(unity)):
                    unity[k][1] = probability_low
                    unity[k][2] = prob_dict[unity[k][0]] * diff + probability_low
                    probability_low = unity[k][2]
                break
    low = 0
    high = 0
    for i in range(len(unity)):
        if unity[i][0] == seq[-1]:
            low = unity[i][1]
            high = unity[i][2]
    point = (low + high) / 2
    size_cod = math.ceil(math.log((1 / (high - low)), 2) + 1)
    bin_code = fl_bins(point, size_cod)
    return [point, alphabet_len, alphabet_list, prob], bin_code


def fl_bins(point, size):
    bin_code = '{:0{}b}'.format(int(point * (2 ** size)), size)
    return bin_code


def dec_ac(encod_data_ac, seq_length):
    point, alphabet_size, alphabet, probability = encod_data_ac
    unity = [[alphabet[i], sum(probability[:i]), sum(probability[:i + 1])] for i in range(alphabet_size)]
    decoded_seq = ""
    for i in range(int(seq_length)):
        for symbol, prob_low, prob_high in unity:
            if prob_low < point < prob_high:
                diff = prob_high - prob_low
                decoded_seq += symbol
                for j in range(alphabet_size):
                    _, prob_l, prob_h = unity[j]
                    unity[j][1], unity[j][2] = prob_low, probability[j] * diff + prob_low
                    prob_low = unity[j][2]
                break
    return decoded_seq


def enc_ch(uniq_chars, prob, seq):
    alphabet = list(uniq_chars)
    prob = [prob[symbol] for symbol in alphabet]
    final_list = [[alphabet[i], prob[i]] for i in range(len(alphabet))]
    final_list.sort(key=lambda x: x[1])

    if 1 in prob and len(set(prob)) == 1:
        symbol_code = [[alphabet[i], "1" * i + "0"] for i in range(len(alphabet))]
        encode = "".join([symbol_code[alphabet.index(c)][1] for c in seq])
    else:
        tree = []
        for _ in range(len(final_list) - 1):
            left = final_list.pop(0)
            right = final_list.pop(0)
            tot = left[1] + right[1]
            tree.append([left[0], right[0]])
            final_list.append([left[0] + right[0], tot])
            final_list.sort(key=lambda x: x[1])

        tree.reverse()
        alphabet.sort()
        symbol_code = []

        for i in range(len(alphabet)):
            code = ""
            for j in range(len(tree)):
                if alphabet[i] in tree[j][0]:
                    code += '0'
                    if alphabet[i] == tree[j][0]:
                        break
                else:
                    code += '1'
                    if alphabet[i] == tree[j][1]:
                        break
            symbol_code.append([alphabet[i], code])

        encode = "".join([symbol_code[i][1] for i in range(len(alphabet)) if symbol_code[i][0] == c][0] for c in seq)

    return [encode, symbol_code], encode


def dec_ch(encoded_seq, seq):
    encode = list(encoded_seq[0])
    symbol_code = encoded_seq[1]
    count = 0
    flag: bool = False

    for i in range(len(encode)):
        for j in range(len(symbol_code)):
            if encode[i] == symbol_code[j][1]:
                seq += str(symbol_code)
                flag = True

        if flag is True:
            flag = False
        else:
            count += 1

            if count == len(encode):
                break
            else:
                encode.insert(i + 1, str(encode[i] + encode[i + 1]))
                encode.pop(i + 2)

    return seq


def start():
    file = "results_AC_CH.txt"

    create_file(file)
    results = []
    length = 12
    border = "-"*100

    with open("sequence.txt", "r", encoding="utf-8") as ac_ch:
        original = ast.literal_eval(ac_ch.read())
        ac_ch.close()

    for seq in original:
        seq = seq[:length]
        sequence_length = len(seq)
        unique_chars = set(seq)
        sequence_alphabet_size = len(unique_chars)
        counts = collections.Counter(seq)
        probability = {symbol: count / sequence_length for symbol, count in counts.items()}
        entropy = -sum(p * math.log2(p) for p in probability.values())
        append_file(file, f'Оригінальна послідовність: {seq}')
        append_file(file, f'Ентропія: {entropy}')
        append_file(file, border)
        encoded_data_ac, encoded_sequence_ac = enc_ac(unique_chars, probability, sequence_alphabet_size, seq)
        bps_ac = round(len(encoded_sequence_ac) / sequence_length, 2)
        decoded_sequence_ac = dec_ac(encoded_data_ac, sequence_length)
        encoded_data_hc, encoded_sequence_hc = enc_ch(unique_chars, probability, seq)
        bps_hc = round(len(encoded_sequence_hc) / sequence_length, 2)
        decoded_sequence_hc = dec_ch(encoded_data_hc, seq)
        append_file(file, 'Арифметичне кодування')
        append_file(file, f'Дані закодованої АС послідовності: {encoded_data_ac}')
        append_file(file, f'Закодована АС послідовність: {encoded_sequence_ac}')
        append_file(file, f'Значення bps при кодуванні АС: {bps_ac}')
        append_file(file, f'Декодована АС послідовність: {decoded_sequence_ac}')
        append_file(file, border)
        append_file(file, 'Кодування Хаффмана')
        append_file(file, f'Дані закодованої HС послідовності: {encoded_data_hc}')
        append_file(file, f'Закодована HС послідовність: {encoded_sequence_hc}')
        append_file(file, f'Значення bps при кодуванні HС: {bps_hc}')
        append_file(file, f'Декодована HС послідовність: {decoded_sequence_hc}')
        append_file(file, border)
        results.append(
            [round(entropy, 2),
             bps_ac,
             bps_hc])

    fig, ax = plt.subplots(figsize=(14 / 1.54, 8 / 1.54))
    headers = ['Ентропія', 'bps AC', 'bps CH']
    row = ['Послідовність 1', 'Послідовність 2', 'Послідовність 3', 'Послідовність 4', 'Послідовність 5',
           'Послідовність 6', 'Послідовність 7', 'Послідовність 8']
    ax.axis('off')
    table = ax.table(cellText=results, colLabels=headers, rowLabels=row, loc='center', cellLoc='center')
    table.auto_set_font_size(True)
    table.set_fontsize(14)
    table.scale(0.6, 2.2)
    fig.savefig('Результати стиснення методами AC та CH' + '.jpg', dpi=300)


start()
