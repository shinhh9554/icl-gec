import re
from kiwipiepy import Kiwi
from konlpy.tag import  Mecab
from diff_match_patch import diff_match_patch



class ErrorExtraction:
    def __init__(self):
        self.mecab = Mecab()
        self.kiwi = Kiwi()
        self.dmp = diff_match_patch()
        self.DIFF_DELETE = -1
        self.DIFF_INSERT = 1
        self.DIFF_EQUAL = 0
        self.error_pattern_lst = []
        with open('ko_dic.txt', 'r', encoding='utf-8') as f:
            self.dic = [i.strip() for i in f.readlines()]

    def append_error_pattern(self, c_word, e_word, error_tag):
        self.error_pattern_lst.append((e_word, error_tag, c_word))
        if error_tag != '0':
            pattern = re.compile(r'[!.?~]+')
            non_punctuation_c_word = pattern.sub('', c_word)
            non_punctuation_e_word = pattern.sub('', e_word)
            if (non_punctuation_e_word, error_tag, non_punctuation_c_word) not in self.error_pattern_lst:
                self.error_pattern_lst.append((non_punctuation_e_word, error_tag, non_punctuation_c_word))

    def annotate_tag(self, c_word, e_word):
        pattern = re.compile(r'[!"#$%&\'()*+,-./:;<=>?@\[\]^_\`{|}~\\\\]')
        non_punctuation_c_word = pattern.sub('', c_word)
        non_punctuation_e_word = pattern.sub('', e_word)
        if non_punctuation_c_word.strip() == non_punctuation_e_word.strip() and c_word != e_word:
            return '0'

        non_space_c_word = non_punctuation_c_word.replace(' ', '')
        non_space_e_word = non_punctuation_e_word.replace(' ', '')
        if non_space_c_word == non_space_e_word and c_word != e_word:
            return '1'

        word_diff = self.dmp.diff_main(non_space_c_word, non_space_e_word)
        self.dmp.diff_cleanupSemantic(word_diff)
        input_count = 0
        del_count = 0
        for op, _ in word_diff:
            if op == 1:
                input_count += 1
            elif op == -1:
                del_count += 1
		
        if del_count > 0 and input_count == 0:
            return '2'

        if input_count > 0 and del_count == 0:
            return '3'

        # ?????? ????????? ???????????? ????????? ????????? ????????? ??????
        # ????????? ????????? ????????? ??????
        dic_count = 0
        for token in non_punctuation_e_word.split():
            if token in non_punctuation_c_word:
                continue

            if token in self.dic:
                dic_count += 1

        if dic_count > 0:
            return '4'
        else:
            return '5'


    def extract_error(self, correct, error):
        self.error_pattern_lst = []
        diff = self.dmp.diff_main(correct, error)
        self.dmp.diff_cleanupSemantic(diff)

        for idx, (op, data) in enumerate(diff):
            if op == self.DIFF_INSERT:
                if idx == len(diff) - 1:
                    # ?????? ??????
                    # correct: "????????? ?????????." -> error: "????????? ?????????.."
                    if diff[idx - 1][0] == self.DIFF_EQUAL:
                        error_text = diff[idx - 1][1].split()[-1] + data
                        correct_text = diff[idx - 1][1].split()[-1]
                        error_tag = self.annotate_tag(correct_text, error_text)
                        self.append_error_pattern(correct_text, error_text, error_tag)
                    elif diff[idx - 1][0] == self.DIFF_INSERT:
                        error_text = diff[idx - 2][1].split()[-1] + diff[idx - 1][1] + data
                        correct_text = diff[idx - 2][1].split()[-1]
                        error_tag = self.annotate_tag(correct_text, error_text)
                        self.append_error_pattern(correct_text, error_text, error_tag)
                    # ?????? ??????
                    # correct: "????????? ???????????????" -> error: "????????? ???????????????"
                    # correct: "?????? ??????." -> error: "?????? ??????"
                    else:
                        if len(diff) == 2:
                            error_text = data
                            correct_text = diff[idx - 1][1]
                            error_tag = self.annotate_tag(correct_text, error_text)
                            self.append_error_pattern(correct_text, error_text, error_tag)
                        elif diff[idx - 2][1][-1] == ' ':
                            if diff[idx - 2][1] == ' ':
                                error_text = data
                                correct_text = diff[idx - 1][1]
                                error_tag = self.annotate_tag(correct_text, error_text)
                                self.append_error_pattern(correct_text, error_text, error_tag)
                            else:
                                error_text = diff[idx - 2][1].split()[-1] + ' ' + data
                                correct_text = diff[idx - 2][1].split()[-1] + ' ' + diff[idx - 1][1]
                                error_tag = self.annotate_tag(correct_text, error_text)
                                self.append_error_pattern(correct_text, error_text, error_tag)
                        else:
                            error_text = diff[idx - 2][1].split()[-1] + data
                            correct_text = diff[idx - 2][1].split()[-1] + diff[idx - 1][1]
                            error_tag = self.annotate_tag(correct_text, error_text)
                            self.append_error_pattern(correct_text, error_text, error_tag)

            elif op == self.DIFF_DELETE:
                # ?????? ??????
                # correct: "????????? ?????????." -> error: "????????? ?????????"
                if idx == len(diff) - 1:
                    if diff[idx - 1][0] == self.DIFF_EQUAL:
                        error_text = diff[idx - 1][1].split()[-1]
                        correct_text = diff[idx - 1][1].split()[-1] + data
                        error_tag = self.annotate_tag(correct_text, error_text)
                        self.append_error_pattern(correct_text, error_text, error_tag)

            elif op == self.DIFF_EQUAL:
                if idx == 0:
                    continue

                if diff[idx - 1][0] == self.DIFF_DELETE:
                    # ???????????? ?????? ??????
                    if diff[idx - 1][1] == ' ':
                        if idx - 2 >= 0:
                            if diff[idx - 2][1][-1] != ' ':
                                if data[0] == ' ':
                                    error_text = diff[idx - 2][1].split()[-1] + ' ' + data.split()[0]
                                    correct_text = diff[idx - 2][1].split()[-1] + diff[idx - 1][1] + ' ' + data.split()[
                                        0]
                                    error_tag = self.annotate_tag(correct_text, error_text)
                                    self.append_error_pattern(correct_text, error_text, error_tag)
                                elif diff[idx - 2][0] == self.DIFF_DELETE:
                                    error_text = diff[idx - 3][1].split()[-1] + data.split()[0]
                                    correct_text = diff[idx - 3][1].split()[-1] + diff[idx - 2][1] + diff[idx - 1][1] + \
                                                   data.split()[0]
                                    error_tag = self.annotate_tag(correct_text, error_text)
                                    self.append_error_pattern(correct_text, error_text, error_tag)
                                else:
                                    error_text = diff[idx - 2][1].split()[-1] + data.split()[0]
                                    correct_text = diff[idx - 2][1].split()[-1] + diff[idx - 1][1] + data.split()[0]
                                    error_tag = self.annotate_tag(correct_text, error_text)
                                    self.append_error_pattern(correct_text, error_text, error_tag)
                            else:
                                error_text = diff[idx - 2][1].split()[-1] + ' ' + data.split()[0]
                                correct_text = diff[idx - 2][1].split()[-1] + ' ' + diff[idx - 1][1] + data.split()[0]
                                error_tag = self.annotate_tag(correct_text, error_text)
                                self.append_error_pattern(correct_text, error_text, error_tag)
                        else:
                            error_text = data.split()[0]
                            correct_text = diff[idx - 1][1] + data.split()[0]
                            error_tag = self.annotate_tag(correct_text, error_text)
                            self.append_error_pattern(correct_text, error_text, error_tag)
                    else:
                        if idx - 2 >= 0:
                            if diff[idx - 2][1][-1] != ' ':
                                # ?????? ??????
                                # correct: "?????? ????????? ????????? ??????." -> error: "?????? ?????? ????????? ??????."
                                # correct: "??????! ????????????" -> error: "?????? ????????????"
                                if data[0] == ' ':
                                    error_text = diff[idx - 2][1].split()[-1] + ' '
                                    correct_text = diff[idx - 2][1].split()[-1] + diff[idx - 1][1] + ' '
                                    error_tag = self.annotate_tag(correct_text, error_text)
                                    self.append_error_pattern(correct_text, error_text, error_tag)
                                elif diff[idx - 1][1][-1] == ' ':
                                    error_text = diff[idx - 2][1].split()[-1] + data.split()[0]
                                    correct_text = diff[idx - 2][1].split()[-1] + diff[idx - 1][1] + data.split()[0]
                                    error_tag = self.annotate_tag(correct_text, error_text)
                                    self.append_error_pattern(correct_text, error_text, error_tag)
                                # ?????? ??????
                                # correct: "?????? ???????????? ?????????." -> error: "?????? ????????? ?????????."
                                else:
                                    error_text = diff[idx - 2][1].split()[-1] + data.split()[0]
                                    correct_text = diff[idx - 2][1].split()[-1] + diff[idx - 1][1] + data.split()[0]
                                    error_tag = self.annotate_tag(correct_text, error_text)
                                    self.append_error_pattern(correct_text, error_text, error_tag)
                            else:
                                # ?????? ??????
                                # correct: "????????? ??? ??????." -> error: "?????????  ??????."
                                # correct: "????????? ??? ??????????" -> error: "????????? ??????????"
                                if data[0] == ' ' or diff[idx - 1][1][-1] == ' ':
                                    if ' ' == diff[idx - 2][1]:
                                        error_text = ' '
                                        correct_text = ' ' + diff[idx - 1][1]
                                    else:
                                        error_text = diff[idx - 2][1].split()[-1] + ' '
                                        correct_text = diff[idx - 2][1].split()[-1] + ' ' + diff[idx - 1][1]
                                    error_tag = self.annotate_tag(correct_text, error_text)
                                    self.append_error_pattern(correct_text, error_text, error_tag)
                                # ?????? ??????
                                # correct: "?????? ??????????????? ?????????." -> error: "?????? ???????????? ?????????."
                                else:
                                    error_text = data.split()[0]
                                    correct_text = diff[idx - 1][1] + data.split()[0]
                                    error_tag = self.annotate_tag(correct_text, error_text)
                                    self.append_error_pattern(correct_text, error_text, error_tag)

                        else:
                            # ?????? ??????
                            # correct: "??? ???????????? ?????? ?????? ??? ?????????." -> error: " ???????????? ?????? ?????? ??? ?????????."
                            if data[0] == ' ':
                                error_text = ' ' + data.split()[0]
                                correct_text = diff[idx - 1][1] + ' ' + data.split()[0]
                                error_tag = self.annotate_tag(correct_text, error_text)
                                self.append_error_pattern(correct_text, error_text, error_tag)
                            # ?????? ??????
                            # correct: "??? ???????????? ?????? ?????? ??? ?????????." -> error: "???????????? ?????? ?????? ??? ?????????."
                            elif diff[idx - 1][1][-1] == ' ':
                                error_text = data.split()[0]
                                correct_text = diff[idx - 1][1] + data.split()[0]
                                error_tag = self.annotate_tag(correct_text, error_text)
                                self.append_error_pattern(correct_text, error_text, error_tag)
                            # ?????? ??????
                            # correct: "????????? ??????" -> error: "?????? ??????"
                            else:
                                error_text = data.split()[0]
                                correct_text = diff[idx - 1][1].split()[-1] + data.split()[0]
                                error_tag = self.annotate_tag(correct_text, error_text)
                                self.append_error_pattern(correct_text, error_text, error_tag)

                elif diff[idx - 1][0] == self.DIFF_INSERT:
                    # ???????????? ?????? ??????
                    if diff[idx - 1][1] == ' ' and diff[idx - 1][0] == 0:
                        error_text = diff[idx - 2][1].split()[-1] + diff[idx - 1][1] + data.split()[0]
                        correct_text = diff[idx - 2][1].split()[-1] + data.split()[0]
                        error_tag = self.annotate_tag(correct_text, error_text)
                        self.append_error_pattern(correct_text, error_text, error_tag)
                    else:
                        if idx - 2 >= 0:
                            if diff[idx - 2][0] == self.DIFF_EQUAL:
                                if diff[idx - 2][1][-1] != ' ':
                                    # ?????? ??????
                                    # correct: "?????? ????????? ?????????." -> error: "????????? ????????? ?????????."
                                    # correct: "?????? ????????? ?????????." -> error: "?????? ????????????  ?????????."
                                    if data[0] == ' ':
                                        error_text = diff[idx - 2][1].split()[-1] + diff[idx - 1][1] + ' '
                                        correct_text = diff[idx - 2][1].split()[-1] + ' '
                                        error_tag = self.annotate_tag(correct_text, error_text)
                                        self.append_error_pattern(correct_text, error_text, error_tag)
                                    elif diff[idx - 1][1][-1] == ' ':
                                        error_text = diff[idx - 2][1].split()[-1] + diff[idx - 1][1] + data.split()[0]
                                        correct_text = diff[idx - 2][1].split()[-1] + data.split()[0]
                                        error_tag = self.annotate_tag(correct_text, error_text)
                                        self.append_error_pattern(correct_text, error_text, error_tag)
                                    # ?????? ??????
                                    # correct: "????????? ?????????." -> error: "????????? ????????????."
                                    else:
                                        error_text = diff[idx - 2][1].split()[-1] + diff[idx - 1][1] + data.split()[0]
                                        correct_text = diff[idx - 2][1].split()[-1] + data.split()[0]
                                        error_tag = self.annotate_tag(correct_text, error_text)
                                        self.append_error_pattern(correct_text, error_text, error_tag)
                                else:
                                    # ?????? ??????
                                    # correct: "?????? ???????????? ?????????." -> error: "?????? ???????????? ??? ?????????."
                                    if data[0] == ' ' or diff[idx - 1][1][-1] == ' ':
                                        error_text = diff[idx - 2][1].split()[-1] + ' ' + diff[idx - 1][1]
                                        correct_text = diff[idx - 2][1].split()[-1] + ' '
                                        error_tag = self.annotate_tag(correct_text, error_text)
                                        self.append_error_pattern(correct_text, error_text, error_tag)
                                    # ?????? ??????
                                    # correct: "?????? ??????????????? ?????????." -> error: "?????? ?????????????????? ?????????."
                                    else:
                                        error_text = diff[idx - 1][1] + data.split()[0]
                                        correct_text = data.split()[0]
                                        error_tag = self.annotate_tag(correct_text, error_text)
                                        self.append_error_pattern(correct_text, error_text, error_tag)
                                        # ?????? ??????
                                        # correct: "?????? ??????" -> error: "????????? ??????"
                            elif diff[idx - 2][0] == self.DIFF_DELETE:
                                if idx - 3 >= 0:
                                    if diff[idx - 3][1][-1] != ' ':
                                        # ?????? ??????
                                        # correct: "????????? ???" -> error: "????????? ???"
                                        # correct: "???????????? ???" -> error: "??????, ??????. ???"
                                        if data[0] == ' ':
                                            error_text = diff[idx - 3][1].split()[-1] + diff[idx - 1][1]
                                            correct_text = diff[idx - 3][1].split()[-1] + diff[idx - 2][1]
                                            error_tag = self.annotate_tag(correct_text, error_text)
                                            self.append_error_pattern(correct_text, error_text, error_tag)

                                        else:
                                            # ?????? ??????
                                            # correct: "???????????? ??????(?) ?????? ?????? ?????????." -> error: "???????????? ????????????? ???????????????"
                                            if diff[idx - 2][1][-1] == ' ':
                                                if data[0] == ' ':
                                                    error_text = diff[idx - 3][1].split()[-1] + diff[idx - 1][1]
                                                    correct_text = diff[idx - 3][1].split()[-1] + diff[idx - 2][1]
                                                    error_tag = self.annotate_tag(correct_text, error_text)
                                                    self.append_error_pattern(correct_text, error_text, error_tag)
                                                else:
                                                    error_text = diff[idx - 3][1].split()[-1] + diff[idx - 1][1] + \
                                                                 data.split()[0]
                                                    correct_text = diff[idx - 3][1].split()[-1] + diff[idx - 2][1] + \
                                                                   data.split()[0]
                                                    error_tag = self.annotate_tag(correct_text, error_text)
                                                    self.append_error_pattern(correct_text, error_text, error_tag)
                                            # ?????? ??????
                                            # correct: "????????? ????????????." -> error: "????????? ?????????."
                                            else:
                                                if diff[idx - 2][1][0] == ' ':
                                                    error_text = diff[idx - 1][1] + data.split()[0]
                                                    correct_text = diff[idx - 2][1] + data.split()[0]
                                                    error_tag = self.annotate_tag(correct_text, error_text)
                                                    self.append_error_pattern(correct_text, error_text, error_tag)
                                                else:
                                                    error_text = diff[idx - 3][1].split()[-1] + diff[idx - 1][
                                                        1] + data.split()[0]
                                                    correct_text = diff[idx - 3][1].split()[-1] + diff[idx - 2][
                                                        1] + data.split()[0]
                                                    error_tag = self.annotate_tag(correct_text, error_text)
                                                    self.append_error_pattern(correct_text, error_text, error_tag)
                                    else:
                                        # ?????? ??????
                                        # correct: "?????? ??? ???????????? ???" -> error: "?????? ??? ???????????? ???"
                                        if data[0] == ' ':
                                            error_text = diff[idx - 1][1] + ' ' + data.split()[0]
                                            correct_text = diff[idx - 2][1] + ' ' + data.split()[0]
                                            error_tag = self.annotate_tag(correct_text, error_text)
                                            self.append_error_pattern(correct_text, error_text, error_tag)
                                        # ?????? ??????
                                        # correct: "??? ?????? ?????? ????????????." -> error: "??? ?????? ?????? ????????????."
                                        else:
                                            if diff[idx - 3][1] == ' ':
                                                error_text = diff[idx - 3][1] + diff[idx - 1][1] + data.split()[0]
                                                correct_text = diff[idx - 2][1] + data.split()[0]
                                                error_tag = self.annotate_tag(correct_text, error_text)
                                                self.append_error_pattern(correct_text, error_text, error_tag)
                                            elif diff[idx - 3][1][-1] == ' ':
                                                error_text = diff[idx - 3][1].split()[-1] + ' ' + diff[idx - 1][1] + \
                                                             data.split()[0]
                                                correct_text = diff[idx - 3][1].split()[-1] + ' ' + diff[idx - 2][1] + \
                                                               data.split()[0]
                                                error_tag = self.annotate_tag(correct_text, error_text)
                                                self.append_error_pattern(correct_text, error_text, error_tag)
                                            else:
                                                error_text = diff[idx - 3][1].split()[-1] + diff[idx - 1][1] + \
                                                             data.split()[0]
                                                correct_text = diff[idx - 3][1].split()[-1] + diff[idx - 2][1] + \
                                                               data.split()[0]
                                                error_tag = self.annotate_tag(correct_text, error_text)
                                                self.append_error_pattern(correct_text, error_text, error_tag)
                                else:
                                    # ?????? ??????
                                    # correct: "??? ????????? ????????????." -> error: "??? ????????? ????????????."
                                    if data[0] == ' ':
                                        error_text = diff[idx - 1][1] + ' ' + data.split()[0]
                                        correct_text = diff[idx - 2][1] + ' ' + data.split()[0]
                                        error_tag = self.annotate_tag(correct_text, error_text)
                                        self.append_error_pattern(correct_text, error_text, error_tag)
                                    else:
                                        error_text = diff[idx - 1][1] + data.split()[0]
                                        correct_text = diff[idx - 2][1] + data.split()[0]
                                        error_tag = self.annotate_tag(correct_text, error_text)
                                        self.append_error_pattern(correct_text, error_text, error_tag)

                        else:
                            # ?????? ??????
                            # correct: " 1. ?????? ??????" -> error: ". 1. ?????? ??????"
                            if data[0] == ' ':
                                error_text = diff[idx - 1][1] + ' ' + data.split()[0]
                                correct_text = ' ' + data.split()[0]
                                error_tag = self.annotate_tag(correct_text, error_text)
                                self.append_error_pattern(correct_text, error_text, error_tag)
                            # ?????? ??????
                            # correct: "?????? ??? ????????????" -> error: "??? ?????? ??? ????????????"
                            elif diff[idx - 1][1][-1] == ' ':
                                error_text = diff[idx - 1][1] + data.split()[0]
                                correct_text = data.split()[0]
                                error_tag = self.annotate_tag(correct_text, error_text)
                                self.append_error_pattern(correct_text, error_text, error_tag)
                            # ?????? ??????
                            # correct: "?????? ??? ????????????" -> "????????? ??? ????????????"
                            else:
                                error_text = diff[idx - 1][1] + data.split()[0]
                                correct_text = data.split()[0]
                                error_tag = self.annotate_tag(correct_text, error_text)
                                self.append_error_pattern(correct_text, error_text, error_tag)

        return self.error_pattern_lst

if __name__ == '__main__':
    e = '#???????????? .'
    c = '#????????????'

    error_class = ErrorExtraction()
    a = error_class.extract_error(c, e)

    print()

