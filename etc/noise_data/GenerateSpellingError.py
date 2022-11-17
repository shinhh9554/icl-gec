import re
import hgtk
import pandas as pd
from konlpy.tag import Mecab
from func import annotate, typo_1, typo_2, typo_3, typo_4, typo_5, typo_6, typo_7, typo_8, typo_9, typo_10, typo_11, \
    typo_12, typo_13, typo_14, typo_15, typo_16, typo_17, typo_18, typo_19, typo_20, typo_21, typo_22, typo_23, typo_24, \
    typo_25, typo_26

mecab = Mecab()


def generate_noise(sentence, error_dict):
    keys = error_dict.keys()

    target_list = [key for key in keys if key in sentence]
    if len(target_list) == 0:
        return sentence

    for target in target_list:
        sentence = sentence.replace(target, error_dict[target])

    return sentence


# error dictionary 생성
def generate_error_dict(path):
    data = pd.read_csv(path, header=None)
    error_dict = {}
    for value in data.values:
        error_dict[value[0]] = value[1]

    return error_dict


def generate_spelling_error(sentence):
    error_dict = generate_error_dict('error_list.txt')
    sentence = annotate(sentence, mecab)
    if re.findall('며칠', sentence):
        noise_data = typo_1(sentence)
        if noise_data is not None:
            sentence = noise_data

    if re.findall('[가-힣]+율', sentence) or re.findall('[가-힣]+률', sentence):
        noise_data = typo_2(sentence)
        if noise_data is not None:
            sentence = noise_data

    if re.findall('[가-힣]+열', sentence) or re.findall('[가-힣]+렬', sentence):
        noise_data = typo_3(sentence)
        if noise_data is not None:
            sentence = noise_data

    if re.findall('/SJ', sentence):
        noise_data = typo_4(sentence)
        if noise_data is not None:
            sentence = noise_data

    if re.findall('/HM', sentence):
        noise_data = typo_5(sentence)
        if noise_data is not None:
            sentence = noise_data

    if re.findall('습니', sentence) or re.findall('읍니', sentence):
        noise_data = typo_6(sentence)
        if noise_data is not None:
            sentence = noise_data

    if re.findall('시오', sentence):
        noise_data = typo_7(sentence)
        if noise_data is not None:
            sentence = noise_data

    if re.findall('/AM', sentence) or re.findall('/AV', sentence):
        noise_data = typo_8(sentence)
        if noise_data is not None:
            sentence = noise_data

    if re.findall('/DV', sentence):
        noise_data = typo_9(sentence)
        if noise_data is not None:
            sentence = noise_data

    if re.findall('왠', sentence) or re.findall('웬', sentence):
        noise_data = typo_10(sentence)
        if noise_data is not None:
            sentence = noise_data

    if re.findall('대로', sentence) or re.findall('데로', sentence):
        noise_data = typo_11(sentence)
        if noise_data is not None:
            sentence = noise_data

    if re.findall('어떡', sentence) or re.findall('어떻', sentence):
        noise_data = typo_12(sentence)
        if noise_data is not None:
            sentence = noise_data

    if re.findall('던지', sentence) or re.findall('든지', sentence):
        noise_data = typo_13(sentence)
        if noise_data is not None:
            sentence = noise_data

    if re.findall('다[차-칳]', sentence) or re.findall('닫[하-힣]', sentence):
        noise_data = typo_14(sentence)
        if noise_data is not None:
            sentence = noise_data

    if re.findall('/NV', sentence):
        noise_data = typo_15(sentence)
        if noise_data is not None:
            sentence = noise_data

    if re.findall('반드시', sentence) or re.findall('반듯이', sentence):
        noise_data = typo_16(sentence)
        if noise_data is not None:
            sentence = noise_data

    if re.findall('ㅂㅜㅌᴥㅇ', hgtk.text.decompose(sentence)) \
            or re.findall('ㅂㅜᴥㅊ', hgtk.text.decompose(sentence)):
        noise_data = typo_17(sentence)
        if noise_data is not None:
            sentence = noise_data

    if re.findall('ㄱㅏᴥㄹㅡᴥㅊ', hgtk.text.decompose(sentence)) \
            or re.findall('ㄱㅏᴥㄹㅣᴥㅋ', hgtk.text.decompose(sentence)):
        noise_data = typo_18(sentence)
        if noise_data is not None:
            sentence = noise_data
        noise_data = typo_19(sentence)
        if noise_data is not None:
            sentence = noise_data

    if re.findall('개수', sentence):
        noise_data = typo_20(sentence)
        if noise_data is not None:
            sentence = noise_data

    if re.findall('를/JKO', sentence) or re.findall('을/JKO', sentence):
        noise_data = typo_21(sentence)
        if noise_data is not None:
            sentence = noise_data

    if re.findall('끼어들', sentence):
        noise_data = typo_22(sentence)
        if noise_data is not None:
            sentence = noise_data

    if re.findall('잘/JV', sentence):
        noise_data = typo_23(sentence)
        if noise_data is not None:
            sentence = noise_data

    if re.findall('/MV', sentence):
        noise_data = typo_24(sentence)
        if noise_data is not None:
            sentence = noise_data

    if re.findall('/JX', sentence):
        noise_data = typo_25(sentence)
        if noise_data is not None:
            sentence = noise_data

    if re.findall('/JKX', sentence):
        noise_data = typo_26(sentence)
        if noise_data is not None:
            sentence = noise_data

    noise_data = re.sub("/[A-Z]+", "", sentence)
    noise_data = generate_noise(noise_data, error_dict)

    return noise_data


if __name__ == '__main__':
    a = "너는 오늘 왠지 정말이지."
    c = generate_spelling_error(a)

    print()
