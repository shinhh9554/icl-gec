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

        # 자체 사전을 구축하여 사전에 있으면 맞춤법 오류
        # 사전에 없으면 표준어 오류
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
                    # 추가 오류
                    # correct: "예전에 있었다." -> error: "예전에 있었다.."
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
                    # 교체 오류
                    # correct: "허준도 재미있었어" -> error: "허준도 재미있었더"
                    # correct: "밥을 먹다." -> error: "먹다 밥을"
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
                # 삭제 오류
                # correct: "예전에 있었다." -> error: "예전에 있었다"
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
                    # 띄어쓰기 삭제 오류
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
                                # 삭제 오류
                                # correct: "나는 예전에 미국을 갔다." -> error: "나는 예전 미국을 갔다."
                                # correct: "잠깐! ㅋㅋㅋㅋ" -> error: "잠깐 ㅋㅋㅋㅋ"
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
                                # 삭제 오류
                                # correct: "나는 프랑스에 있었다." -> error: "나는 프스에 있었다."
                                else:
                                    error_text = diff[idx - 2][1].split()[-1] + data.split()[0]
                                    correct_text = diff[idx - 2][1].split()[-1] + diff[idx - 1][1] + data.split()[0]
                                    error_tag = self.annotate_tag(correct_text, error_text)
                                    self.append_error_pattern(correct_text, error_text, error_tag)
                            else:
                                # 삭제 오류
                                # correct: "일어날 수 있다." -> error: "일어날  있다."
                                # correct: "일어날 수 있을까?" -> error: "일어날 있을까?"
                                if data[0] == ' ' or diff[idx - 1][1][-1] == ' ':
                                    if ' ' == diff[idx - 2][1]:
                                        error_text = ' '
                                        correct_text = ' ' + diff[idx - 1][1]
                                    else:
                                        error_text = diff[idx - 2][1].split()[-1] + ' '
                                        correct_text = diff[idx - 2][1].split()[-1] + ' ' + diff[idx - 1][1]
                                    error_tag = self.annotate_tag(correct_text, error_text)
                                    self.append_error_pattern(correct_text, error_text, error_tag)
                                # 삭제 오류
                                # correct: "나는 김포시에서 일했다." -> error: "나는 포시에서 일했다."
                                else:
                                    error_text = data.split()[0]
                                    correct_text = diff[idx - 1][1] + data.split()[0]
                                    error_tag = self.annotate_tag(correct_text, error_text)
                                    self.append_error_pattern(correct_text, error_text, error_tag)

                        else:
                            # 삭제 오류
                            # correct: "이 기세라면 따라 잡을 수 있겠어." -> error: " 기세라면 따라 잡을 수 있겠어."
                            if data[0] == ' ':
                                error_text = ' ' + data.split()[0]
                                correct_text = diff[idx - 1][1] + ' ' + data.split()[0]
                                error_tag = self.annotate_tag(correct_text, error_text)
                                self.append_error_pattern(correct_text, error_text, error_tag)
                            # 삭제 오류
                            # correct: "이 기세라면 따라 잡을 수 있겠어." -> error: "기세라면 따라 잡을 수 있겠어."
                            elif diff[idx - 1][1][-1] == ' ':
                                error_text = data.split()[0]
                                correct_text = diff[idx - 1][1] + data.split()[0]
                                error_tag = self.annotate_tag(correct_text, error_text)
                                self.append_error_pattern(correct_text, error_text, error_tag)
                            # 삭제 오류
                            # correct: "은은한 달빛" -> error: "은한 달빛"
                            else:
                                error_text = data.split()[0]
                                correct_text = diff[idx - 1][1].split()[-1] + data.split()[0]
                                error_tag = self.annotate_tag(correct_text, error_text)
                                self.append_error_pattern(correct_text, error_text, error_tag)

                elif diff[idx - 1][0] == self.DIFF_INSERT:
                    # 띄어쓰기 추가 오류
                    if diff[idx - 1][1] == ' ' and diff[idx - 1][0] == 0:
                        error_text = diff[idx - 2][1].split()[-1] + diff[idx - 1][1] + data.split()[0]
                        correct_text = diff[idx - 2][1].split()[-1] + data.split()[0]
                        error_tag = self.annotate_tag(correct_text, error_text)
                        self.append_error_pattern(correct_text, error_text, error_tag)
                    else:
                        if idx - 2 >= 0:
                            if diff[idx - 2][0] == self.DIFF_EQUAL:
                                if diff[idx - 2][1][-1] != ' ':
                                    # 추가 오류
                                    # correct: "나는 축구를 즐겨봐." -> error: "나는느 축구를 즐겨봐."
                                    # correct: "나는 야구를 즐겨봐." -> error: "나는 야구를르  즐겨봐."
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
                                    # 추가 오류
                                    # correct: "예전에 있었다." -> error: "예전에 있었었다."
                                    else:
                                        error_text = diff[idx - 2][1].split()[-1] + diff[idx - 1][1] + data.split()[0]
                                        correct_text = diff[idx - 2][1].split()[-1] + data.split()[0]
                                        error_tag = self.annotate_tag(correct_text, error_text)
                                        self.append_error_pattern(correct_text, error_text, error_tag)
                                else:
                                    # 추가 오류
                                    # correct: "나는 안양에서 일한다." -> error: "나는 안양에서 서 일한다."
                                    if data[0] == ' ' or diff[idx - 1][1][-1] == ' ':
                                        error_text = diff[idx - 2][1].split()[-1] + ' ' + diff[idx - 1][1]
                                        correct_text = diff[idx - 2][1].split()[-1] + ' '
                                        error_tag = self.annotate_tag(correct_text, error_text)
                                        self.append_error_pattern(correct_text, error_text, error_tag)
                                    # 추가 오류
                                    # correct: "나는 김포시에서 일했다." -> error: "나는 ㄱ김포시에서 일했다."
                                    else:
                                        error_text = diff[idx - 1][1] + data.split()[0]
                                        correct_text = data.split()[0]
                                        error_tag = self.annotate_tag(correct_text, error_text)
                                        self.append_error_pattern(correct_text, error_text, error_tag)
                                        # 교체 오류
                                        # correct: "한시 일분" -> error: "하나시 일분"
                            elif diff[idx - 2][0] == self.DIFF_DELETE:
                                if idx - 3 >= 0:
                                    if diff[idx - 3][1][-1] != ' ':
                                        # 교체 오류
                                        # correct: "가르쳐 줘" -> error: "가르켜 줘"
                                        # correct: "맞어마정 ㅠ" -> error: "맞아, 맞아. ㅠ"
                                        if data[0] == ' ':
                                            error_text = diff[idx - 3][1].split()[-1] + diff[idx - 1][1]
                                            correct_text = diff[idx - 3][1].split()[-1] + diff[idx - 2][1]
                                            error_tag = self.annotate_tag(correct_text, error_text)
                                            self.append_error_pattern(correct_text, error_text, error_tag)

                                        else:
                                            # 교체 오류
                                            # correct: "국내에서 온천(?) 갈까 생각 중이야." -> error: "국내에서 온천?갈까 생각중이야"
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
                                            # 교체 오류
                                            # correct: "의자가 부셔졌다." -> error: "의자가 부셨다."
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
                                        # 교체 오류
                                        # correct: "이제 곧 들어가야 해" -> error: "이제 곶 들어가야 해"
                                        if data[0] == ' ':
                                            error_text = diff[idx - 1][1] + ' ' + data.split()[0]
                                            correct_text = diff[idx - 2][1] + ' ' + data.split()[0]
                                            error_tag = self.annotate_tag(correct_text, error_text)
                                            self.append_error_pattern(correct_text, error_text, error_tag)
                                        # 교체 오류
                                        # correct: "이 길로 곧장 가야한다." -> error: "이 길로 곶장 가야한다."
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
                                    # 교체 오류
                                    # correct: "곧 경기가 시작한다." -> error: "곶 경기가 시작한다."
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
                            # 추가 오류
                            # correct: " 1. 연구 목적" -> error: ". 1. 연구 목적"
                            if data[0] == ' ':
                                error_text = diff[idx - 1][1] + ' ' + data.split()[0]
                                correct_text = ' ' + data.split()[0]
                                error_tag = self.annotate_tag(correct_text, error_text)
                                self.append_error_pattern(correct_text, error_text, error_tag)
                            # 추가 오류
                            # correct: "지금 이 기세라면" -> error: "ㅈ 지금 이 기세라면"
                            elif diff[idx - 1][1][-1] == ' ':
                                error_text = diff[idx - 1][1] + data.split()[0]
                                correct_text = data.split()[0]
                                error_tag = self.annotate_tag(correct_text, error_text)
                                self.append_error_pattern(correct_text, error_text, error_tag)
                            # 추가 오류
                            # correct: "지금 이 기세라면" -> "ㅈ지금 이 기세라면"
                            else:
                                error_text = diff[idx - 1][1] + data.split()[0]
                                correct_text = data.split()[0]
                                error_tag = self.annotate_tag(correct_text, error_text)
                                self.append_error_pattern(correct_text, error_text, error_tag)

        return self.error_pattern_lst

if __name__ == '__main__':
    e = '#아프지마 .'
    c = '#아프지마'

    error_class = ErrorExtraction()
    a = error_class.extract_error(c, e)

    print()

