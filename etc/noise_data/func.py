import re
import os
import json
from jamo import h2j, j2hcj
import hgtk
from konlpy.tag import Mecab

jongsung_list = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ',
                 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']


# 몇일과 며칠
def typo_1(sentence):
    # noise 생성
    noise_data = re.sub('며칠', '몇 일', sentence)

    return noise_data


# 율과 률
def typo_2(sentence):
    if re.findall('율', sentence):
        noise_data = re.sub('율', '률', sentence)
        return noise_data

    elif re.findall('률', sentence):
        noise_data = re.sub('률', '율', sentence)
        return noise_data


# 열과 렬
def typo_3(sentence):
    if re.findall('열', sentence):
        noise_data = re.sub('열', '렬', sentence)

        return noise_data


    elif re.findall('렬', sentence):
        noise_data = re.sub('렬', '열', sentence)

        return noise_data


# 로서와 로써
def typo_4(sentence):
    if re.findall('로서/SJ', sentence):
        noise_data = re.sub('로서/SJ', '로써', sentence)

        return noise_data


    elif re.findall('로써/SJ', sentence):
        noise_data = re.sub('로써/SJ', '로서', sentence)

        return noise_data


# 이와 히 (ex: 깨끗이/깨끗히)
def typo_5(sentence):
    if re.findall('이/HM', sentence):
        noise_data = re.sub('이/HM', '히', sentence)

        return noise_data

    elif re.findall('히/HM', sentence):
        noise_data = re.sub('히/HM', '이', sentence)

        return noise_data


# 습니다.와 읍니다.
def typo_6(sentence):
    if re.findall('습니', sentence):
        noise_data = re.sub('습니', '읍니', sentence)

        return noise_data

    elif re.findall('읍니', sentence):
        noise_data = re.sub('읍니', '습니', sentence)

        return noise_data


# 시오.와 시요.
def typo_7(sentence):
    noise_data = re.sub('시오', '시요', sentence)

    return noise_data


# 안과 않
def typo_8(sentence):
    if re.findall('/AM', sentence):
        noise_data = re.sub('안/AM', '않', sentence)

        return noise_data


    elif re.findall('/AV', sentence):
        noise_data = re.sub('않/AV', '안', sentence)

        return noise_data

# 되와 돼
def typo_9(sentence):
    if re.findall('[되-됳]/DV', sentence):
        noise_data = hgtk.text.compose(re.sub('ㄷㅚ([ᴥ/ㄱ-ㅎ]+)DV', r'ㄷㅙ\1DV', hgtk.text.decompose(sentence)))
        return noise_data


    elif re.findall('[돼-됗]/DV', sentence):
        noise_data = hgtk.text.compose(re.sub('ㄷㅙ([ᴥ/ㄱ-ㅎ]+)DV', r'ㄷㅚ\1DV', hgtk.text.decompose(sentence)))
        return noise_data


# 왠과 웬
def typo_10(sentence):
    if re.findall('왠', sentence):
        noise_data = re.sub('왠', '웬', sentence)

        return noise_data


    elif re.findall('웬', sentence):
        noise_data = re.sub('웬', '왠', sentence)

        return noise_data


# 대로와 데로
def typo_11(sentence):
    if re.findall('대로', sentence):
        noise_data = re.sub('대로', '데로', sentence)
        return noise_data


    elif re.findall('데로', sentence):
        noise_data = re.sub('데로', '대로', sentence)
        return noise_data


# 어떡과 어떻
def typo_12(sentence):
    if re.findall('어떡', sentence):
        noise_data = re.sub('어떡', '어떻', sentence)
        return noise_data


    elif re.findall('어떻', sentence):
        noise_data = re.sub('어떻', '어떡', sentence)
        return noise_data

# 던지와 든지
def typo_13(sentence):
    if re.findall('든지', sentence):
        noise_data = re.sub('든지', '던지', sentence)
        return noise_data

    elif re.findall('던지', sentence):
        noise_data = re.sub('던지', '든지', sentence)
        return noise_data


# 닫히다와 다치다.
def typo_14(sentence):
    if re.findall('다[차-칳]', sentence):
        noise_data = hgtk.text.compose(re.sub('ㄷㅏᴥㅊ', 'ㄷㅏㄷᴥㅎ', hgtk.text.decompose(sentence)))
        return noise_data

    elif re.findall('닫[하-힣]', sentence):
        noise_data = hgtk.text.compose(re.sub('ㄷㅏㄷᴥㅎ', 'ㄷㅏᴥㅊ', hgtk.text.decompose(sentence)))
        return noise_data


# 나았다와 낳았다.
def typo_15(sentence):
    if re.findall('낳았', sentence):
        noise_data = re.sub('낳았', '나았', sentence)
        return noise_data

    elif re.findall('나았', sentence):
        noise_data = re.sub('나았', '낳았', sentence)
        return noise_data

    elif re.findall('낫다', sentence):
        noise_data = re.sub('낫다', '낳다', sentence)
        return noise_data

    elif re.findall('낳다', sentence):
        noise_data = re.sub('낳다', '낫다', sentence)
        return noise_data

    elif re.findall('나을', sentence):
        noise_data = re.sub('나을', '낳을', sentence)
        return noise_data

    elif re.findall('낳을', sentence):
        noise_data = re.sub('낳을', '나을', sentence)
        return noise_data



# 반드시와 반듯이
def typo_16(sentence):
    if re.findall('반드시', sentence):
        noise_data = re.sub('반드시', '반듯이', sentence)

        return noise_data


    elif re.findall('반듯이', sentence):
        noise_data = re.sub('반듯이', '반드시', sentence)

        return noise_data


# 붙이다와 부치다.
def typo_17(sentence):
    if re.findall('ㅂㅜㅌᴥㅇ', hgtk.text.decompose(sentence)):
        noise_data = hgtk.text.compose(re.sub('ㅂㅜㅌᴥㅇ', 'ㅂㅜᴥㅊ',
                                              hgtk.text.decompose(sentence)))

        return noise_data


    elif re.findall('ㅂㅜᴥㅊ', hgtk.text.decompose(sentence)):
        noise_data = hgtk.text.compose(re.sub('ㅂㅜᴥㅊ', 'ㅂㅜㅌᴥㅇ',
                                              hgtk.text.decompose(sentence)))

        return noise_data


# 가르치다와 가르키다.
def typo_18(sentence):
    if re.findall('ㄱㅏᴥㄹㅡᴥㅊ', hgtk.text.decompose(sentence)):
        noise_data = hgtk.text.compose(re.sub('ㄱㅏᴥㄹㅡᴥㅊ', 'ㄱㅏᴥㄹㅣᴥㅋ',
                                              hgtk.text.decompose(sentence)))

        return noise_data


    elif re.findall('ㄱㅏᴥㄹㅣᴥㅋ', hgtk.text.decompose(sentence)):
        noise_data = re.sub('ㄱㅏᴥㄹㅣᴥㅋ', 'ㄱㅏᴥㄹㅡᴥㅊ', sentence)

        return noise_data


# 가르키다.(비표준어) 교정 - 가르쳤다. 가리켰다.
def typo_19(sentence):
    if re.findall('ㄱㅏᴥㄹㅡᴥㅊ', hgtk.text.decompose(sentence)):
        noise_data = hgtk.text.compose(re.sub('ㄱㅏᴥㄹㅡᴥㅊ', 'ㄱㅏᴥㄹㅡᴥㅋ',
                                              hgtk.text.decompose(sentence)))

        return noise_data


    elif re.findall('ㄱㅏᴥㄹㅣᴥㅋ', hgtk.text.decompose(sentence)):
        noise_data = re.sub('ㄱㅏᴥㄹㅣᴥㅋ', 'ㄱㅏᴥㄹㅡᴥㅋ', sentence)
        return noise_data


# 개수
def typo_20(sentence):

        if re.findall('개수', sentence):
            noise_data = re.sub('개수', '갯수', sentence)

            return noise_data

# 을/를
def typo_21(sentence):
    if re.findall('을', sentence):
        noise_data = re.sub('을/JKO', '를', sentence)

        return noise_data

    elif re.findall('를', sentence):
        noise_data = re.sub('를/JKO', '을', sentence)
        return noise_data


# 끼어들다.
def typo_22(sentence):
    if re.findall('끼어들', sentence):
        noise_data = re.sub('끼어들', '끼여들', sentence)

        return noise_data


# 잘렸다.
def typo_23(sentence):
    if re.findall('잘/JV', sentence):
        noise_data = re.sub('잘/JV', '짤', sentence)

        return noise_data


# 맞히다.
def typo_24(sentence):
    if re.findall('히/MV', sentence):
        noise_data = re.sub('히/MV', '추', sentence)

        return noise_data

    if re.findall('혔/MV', sentence):
        noise_data = re.sub('혔/MV', '췄', sentence)

        return noise_data

    if re.findall('혀/MV', sentence):
        noise_data = re.sub('혀/MV', '춰', sentence)

        return noise_data

    if re.findall('추/MV', sentence):
        noise_data = re.sub('추/MV', '히', sentence)

        return noise_data

    if re.findall('췄/MV', sentence):
        noise_data = re.sub('췄/MV', '혔', sentence)

        return noise_data

    if re.findall('춰/MV', sentence):
        noise_data = re.sub('춰/MV', '혀', sentence)

        return noise_data

    if re.findall('히/MV', sentence):
        noise_data = re.sub('히/MV', '추', sentence)
        return noise_data


def typo_25(sentence):
    if re.findall('은/JX', sentence):
        noise_data = re.sub('은/JX', '는', sentence)

        return noise_data

    elif re.findall('는/JX', sentence):
        noise_data = re.sub('는/JX', '은', sentence)

        return noise_data


def typo_26(sentence):
    if re.findall('이/JKX', sentence):
        noise_data = re.sub('이/JKX', '가', sentence)

        return noise_data

    elif re.findall('가/JKX', sentence):
        noise_data = re.sub('가/JKX', '이', sentence)

        return noise_data


def annotate(string, mecab):
    '''attach pos tags to the given string using Mecab
    mecab: mecab object
    '''
    tokens = mecab.pos(string)
    if string.replace(" ", "") != "".join(token for token, _ in tokens):
        return string
    blanks = [i for i, char in enumerate(string) if char == " "]

    tag_seq = []
    for token, tag in tokens:
        if tag=="NNBC": # bound noun
            tag = "B"
        else:
            tag = tag[0]
        tag_seq.append(tag * (len(token)))
    tag_seq = "".join(tag_seq)

    for i in blanks:
        tag_seq = tag_seq[:i] + " " + tag_seq[i:]

    annotated = ""
    for char, tag in zip(string, tag_seq):
        annotated += char

        if tag == "V" or tag == "E":
            if "ㄷㅚ" in j2hcj(h2j(char)) or "ㄷㅙ" in j2hcj(h2j(char)):
                annotated += "/DV"
            if char == '않':
                annotated += "/AV"
            if re.findall('ㄴㅏ[ᴥㅅㅆ]+[ㄱ-ㅎㅏ-ㅣ]+', hgtk.text.decompose(annotated[-2:])) \
                    or re.findall('ㄴㅏㅎᴥ[ㄱ-ㅎㅏ-ㅣ]+', hgtk.text.decompose(annotated[-2:])):
                annotated += "/NV"
            if char == '잘':
                annotated += '/JV'
            if re.findall('ㅁㅏㅈᴥ[ㄱ-ㅎㅏ-ㅣ]+', hgtk.text.decompose(annotated[-2:])):
                annotated += '/MV'
        if tag == 'M':
            if char == '이' or char == '히':
                annotated += "/HM"
            if char == '안':
                annotated += '/AM'

        if tag == 'J':
            if (char == '서' or char == '써') and annotated[-2] == '로':
                annotated += '/SJ'

            if char == '을' or char == '를':
                annotated += '/JKO'

            if char == '은' or char == '는':
                annotated += '/JX'

            if char == '이' or char == '가':
                annotated += '/JKX'
    return annotated


if __name__ == '__main__':
    mecab = Mecab()
    sentence = '이 연구는 오늘 종료 된다. 깨끗히 끝나다.'
    print(mecab.pos(sentence))

    # print(hgtk.text.compose(re.sub('ㅂㅜㅌᴥㅇ', 'ㅂㅜᴥㅊ',
    #                                               hgtk.text.decompose(sentence))))

