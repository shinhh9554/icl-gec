import re
import random
from g2pk import G2p
from tqdm import tqdm
from pathlib import Path
from GenerateSpellingError import generate_spelling_error

g2p = G2p()


def main(path):
    reg = re.compile(r'[a-zA-Zㄱ-ㅎ가-힣]+')

    with open(path, 'r', encoding='utf-8') as rf1:
        lines = [i.strip() for i in rf1.readlines()]

    noise_dict = {}
    for noise_corpus in tqdm(open('noise_data/new_noise_sample.txt', 'r', encoding='utf-8').readlines()):
        noise_corpus = noise_corpus.replace('\n', '')
        error, correct, label = noise_corpus.split('\t')
        if reg.match(correct):
            if correct in noise_dict.keys():
                error_lst = noise_dict[correct]
                error_lst.append(error)
                noise_dict[correct] = error_lst
            else:
                noise_dict[correct] = [error]
        else:
            continue

    with open(f"noise_data/output_data/noise_sample.tsv", mode='w', encoding='utf-8') as w:
        error_list = []
        for line in tqdm(lines):
            if len(line) > 500:
                continue

            try:
                for noise_key in noise_dict.keys():
                    if noise_key in line:
                        error_word_lst = noise_dict[noise_key]
                        for error_word in error_word_lst:
                            apply_sent = line
                            error_list.append(f"{line}\t{apply_sent.replace(noise_key, error_word)}\n")
            except:
                pass

            try:
                g2p_apply_sent = line
                g2p_error = g2p(g2p_apply_sent)
                error_list.append(f"{line}\t{g2p_error}\n")
            except:
                pass

            try:
                spelling_error_apply_sent = line
                spelling_error = generate_spelling_error(spelling_error_apply_sent)
                error_list.append(f"{line}\t{spelling_error}\n")
            except:
                pass

        w.writelines(error_list)


if __name__ == '__main__':
    input_path = 'noise_data/input_data/sample.txt'
    main(input_path)
