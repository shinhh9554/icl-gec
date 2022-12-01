import re
import json
from tqdm import tqdm
from diff_match_patch import diff_match_patch

except_lst = [
    '밀면 ㅠㅠㅠㅠ 술 해장으로 먹는 밀면, 캬.',
    '아마스빈 버블티에만 있나 보네요. 맛있음. ㅎㅎ 굉장히 맛있음. ㅎㅎ',
    '따따따 따 따 따따따. ㅋㅋㅋ',
    '인생밥집이 1호점이고 옹포리가 인생 밥집 1호점이에요. ㅋㅋㅋㅋ',
    '죽 먹을지 제육볶음 먹을지, 흠. 🤔',
    '푸짐하니 ㅋ ㅋ. 얼 name1 모르냐?',
    '집에 머핀 틀 없고 타르트 틀 케이크 원형(?)은 있다.',
    'ㅋㅋ 그러게요. 옷이 문제인지... 제가 문제인지... ㅋㅋ',
    '#아기화장품 #아기크림 #천연아기화장품 #천연베이비크림 #저자극 아기 화장품.',
    '믿고 쓰는 #아벤느 😍',
    '단단한 것들로 지탱하고 있다고 믿고 있는 삶. 하지만, 어쩌면! 진짜! 작가가 썼던 것처럼 우리의 삶은 `흰 ` 모래와도 같이 부스러지고 날아갈 수 있는 가벼운 게 아닐까. '
    '결국 우리는 모래의 일부가 된다. 이는 부인할 수 없는 사실이자 진리이다. 결국 우리는 희고 가벼운 것이 되고 만다. `그리고 그녀는 자주 잊었다. '
    '자신의 몸이(우리 모두의 몸이) 모래의 집이란 걸. '
    '부스러져 왔으며 부스러지고 있다는 걸. '
    '끈질기게 손가락 사이로 흘러내리고 있다는 걸.` 「모래」 중에서. `여러 해 뒤 그 생명 - 재생 - 부활의 꽃나무들 아래를 지나다 그녀는 생각했다. '
    '그때 왜 우리는 하필 백목련을 골랐을까. 흰 꽃은 생명과 연결되어 있는 걸까, 아니면 죽음과? '
    '인도·유럽어에서 텅 빔과 흰빛, 검음과 불꽃이 모두 같은 어원을 갖는다고 그녀는 읽었다. '
    '어둠을 안고 타오르는 텅 빈 흰 불꽃들ㅡ 그것이 삼월에 짧게 꽃 피는 백목련 두 그루인 걸까? `「백목련」 중에서.',
    '13년 케이맘 #유아물티슈 시작으로 마더케이 유아 물티슈, 비데 물티슈, 케이맘순한아기물티슈 등을 출시하면서…',
    '#올림픽공원 #나홀로나무 랑 나 홀로. ❤👍🏻',
    '#여행가고싶다 혼자라도... 가고 싶다.',
    '달려라 달려라. [무병장수해로 🎎]',
    '퇴근 퇴근 퇴근 퇴근 퇴근시켜주세요. ₍₍ ◝(`ω`◝) ⁾⁾ ₍₍ (◟`ω`)◟ ⁾⁾',
    '박사님... 박사님... (찡얼거리면서 손으로 작게 문을 툭툭거린다.) 열어 주세요, 네?'
]



class ErrorAnnotation:
    def __init__(self):
        self.dmp = diff_match_patch()
        self.DIFF_DELETE = -1
        self.DIFF_INSERT = 1
        self.DIFF_EQUAL = 0

        self.n = 1
        self.annotation = ''
        self.error_label = None
        self.correct_label = None
        self.correction = ''

        self.en = 0
        self.cn = 0

        self.b_e_word = None
        self.b_c_word = None

    def annotate_label_idx(self, e_word, c_word, error, correct):
        if self.b_c_word is not None:
            if c_word in self.b_c_word:
                self.cn += self.b_c_word.index(c_word) + 1

            if e_word in self.b_e_word:
                self.en += self.b_e_word.index(e_word) + 1

        s_ei = error.find(e_word, self.en)
        e_ei = s_ei + len(e_word)
        s_ci = correct.find(c_word, self.cn)
        e_ci = s_ci + len(c_word)
        self.error_label[s_ei:e_ei] = ['E'] * len(e_word)
        self.correct_label[s_ci:e_ci] = ['C'] * len(c_word)

        self.cn = s_ci
        self.en = s_ei
        self.b_c_word = c_word
        self.b_e_word = e_word

    def annotate_error(self, correct, error):
        diff = self.dmp.diff_main(correct, error)
        self.dmp.diff_cleanupSemantic(diff)

        self.en = 0
        self.cn = 0
        self.b_c_word = None
        self.b_e_word = None

        self.annotation = [i for i in error]
        self.correction = []
        self.error_label = ['O'] * len(error)
        self.correct_label = ['O'] * len(correct)

        for idx, (op, data) in enumerate(diff):
            if op == self.DIFF_INSERT:
                if idx == len(diff) - 1:
                    # 추가 오류
                    # correct: "예전에 있었다." -> error: "예전에 있었다.."
                    if diff[idx - 1][0] == self.DIFF_EQUAL:
                        error_text = diff[idx - 1][1].split()[-1] + data
                        correct_text = diff[idx - 1][1].split()[-1]
                        self.annotate_label_idx(error_text, correct_text, error, correct)
                    elif diff[idx - 1][0] == self.DIFF_INSERT:
                        error_text = diff[idx - 2][1].split()[-1] + diff[idx - 1][1] + data
                        correct_text = diff[idx - 2][1].split()[-1]
                        self.annotate_label_idx(error_text, correct_text, error, correct)
                    # 교체 오류
                    # correct: "허준도 재미있었어" -> error: "허준도 재미있었더"
                    # correct: "밥을 먹다." -> error: "먹다 밥을"
                    else:
                        if len(diff) == 2:
                            error_text = data
                            correct_text = diff[idx - 1][1]
                            self.annotate_label_idx(error_text, correct_text, error, correct)
                        elif diff[idx - 2][1][-1] == ' ':
                            if diff[idx - 2][1] == ' ':
                                error_text = data
                                correct_text = diff[idx - 1][1]
                                self.annotate_label_idx(error_text, correct_text, error, correct)
                            else:
                                error_text = diff[idx - 2][1].split()[-1] + ' ' + data
                                correct_text = diff[idx - 2][1].split()[-1] + ' ' + diff[idx - 1][1]
                                self.annotate_label_idx(error_text, correct_text, error, correct)
                        else:
                            error_text = diff[idx - 2][1].split()[-1] + data
                            correct_text = diff[idx - 2][1].split()[-1] + diff[idx - 1][1]
                            self.annotate_label_idx(error_text, correct_text, error, correct)

            elif op == self.DIFF_DELETE:
                # 삭제 오류
                # correct: "예전에 있었다." -> error: "예전에 있었다"
                if idx == len(diff) - 1:
                    if diff[idx - 1][0] == self.DIFF_EQUAL:
                        error_text = diff[idx - 1][1].split()[-1]
                        correct_text = diff[idx - 1][1].split()[-1] + data
                        self.annotate_label_idx(error_text, correct_text, error, correct)

            elif op == self.DIFF_EQUAL:
                if idx == 0:
                    continue

                if diff[idx-1][0] == self.DIFF_DELETE:
                    # 띄어쓰기 삭제 오류
                    if diff[idx-1][1] == ' ':
                        if idx - 2 >= 0:
                            if diff[idx - 2][1][-1] != ' ':
                                if data[0] == ' ':
                                    error_text = diff[idx - 2][1].split()[-1] + ' ' + data.split()[0]
                                    correct_text = diff[idx - 2][1].split()[-1] + diff[idx - 1][1] + ' ' + data.split()[0]
                                    self.annotate_label_idx(error_text, correct_text, error, correct)
                                elif diff[idx - 2][0] == self.DIFF_DELETE:
                                    error_text = diff[idx - 3][1].split()[-1] + data.split()[0]
                                    correct_text = diff[idx - 3][1].split()[-1] + diff[idx - 2][1] + diff[idx - 1][1] + data.split()[0]
                                    self.annotate_label_idx(error_text, correct_text, error, correct)
                                else:
                                    error_text = diff[idx - 2][1].split()[-1] + data.split()[0]
                                    correct_text = diff[idx - 2][1].split()[-1] + diff[idx - 1][1] + data.split()[0]
                                    self.annotate_label_idx(error_text, correct_text, error, correct)
                            else:
                                error_text = diff[idx - 2][1].split()[-1] + ' ' + data.split()[0]
                                correct_text = diff[idx - 2][1].split()[-1] + ' ' + diff[idx - 1][1] + data.split()[0]
                                self.annotate_label_idx(error_text, correct_text, error, correct)
                        else:
                            error_text = data.split()[0]
                            correct_text = diff[idx - 1][1] + data.split()[0]
                            self.annotate_label_idx(error_text, correct_text, error, correct)
                    else:
                        if idx - 2 >= 0:
                            if diff[idx - 2][1][-1] != ' ':
                                # 삭제 오류
                                # correct: "나는 예전에 미국을 갔다." -> error: "나는 예전 미국을 갔다."
                                # correct: "잠깐! ㅋㅋㅋㅋ" -> error: "잠깐 ㅋㅋㅋㅋ"
                                if data[0] == ' ':
                                    error_text = diff[idx - 2][1].split()[-1] + ' '
                                    correct_text = diff[idx - 2][1].split()[-1] + diff[idx - 1][1] + ' '
                                    self.annotate_label_idx(error_text, correct_text, error, correct)
                                elif diff[idx - 1][1][-1] == ' ':
                                    error_text = diff[idx - 2][1].split()[-1] + data.split()[0]
                                    correct_text = diff[idx - 2][1].split()[-1] + diff[idx - 1][1] + data.split()[0]
                                    self.annotate_label_idx(error_text, correct_text, error, correct)
                                # 삭제 오류
                                # correct: "나는 프랑스에 있었다." -> error: "나는 프스에 있었다."
                                else:
                                    error_text = diff[idx - 2][1].split()[-1] + data.split()[0]
                                    correct_text = diff[idx - 2][1].split()[-1] + diff[idx - 1][1] + data.split()[0]
                                    self.annotate_label_idx(error_text, correct_text, error, correct)
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
                                    self.annotate_label_idx(error_text, correct_text, error, correct)
                                # 삭제 오류
                                # correct: "나는 김포시에서 일했다." -> error: "나는 포시에서 일했다."
                                else:
                                    error_text = data.split()[0]
                                    correct_text = diff[idx - 1][1] + data.split()[0]
                                    self.annotate_label_idx(error_text, correct_text, error, correct)

                        else:
                            # 삭제 오류
                            # correct: "이 기세라면 따라 잡을 수 있겠어." -> error: " 기세라면 따라 잡을 수 있겠어."
                            if data[0] == ' ':
                                error_text = ' ' + data.split()[0]
                                correct_text = diff[idx - 1][1] + ' ' + data.split()[0]
                                self.annotate_label_idx(error_text, correct_text, error, correct)
                            # 삭제 오류
                            # correct: "이 기세라면 따라 잡을 수 있겠어." -> error: "기세라면 따라 잡을 수 있겠어."
                            elif diff[idx - 1][1][-1] == ' ':
                                error_text = data.split()[0]
                                correct_text = diff[idx - 1][1] + data.split()[0]
                                self.annotate_label_idx(error_text, correct_text, error, correct)
                            # 삭제 오류
                            # correct: "은은한 달빛" -> error: "은한 달빛"
                            else:
                                error_text = data.split()[0]
                                correct_text = diff[idx - 1][1].split()[-1] + data.split()[0]
                                self.annotate_label_idx(error_text, correct_text, error, correct)

                elif diff[idx-1][0] == self.DIFF_INSERT:
                    # 띄어쓰기 추가 오류
                    if diff[idx - 1][1] == ' ' and diff[idx - 1][0] == 0:
                        error_text = diff[idx - 2][1].split()[-1] + diff[idx - 1][1] + data.split()[0]
                        correct_text = diff[idx - 2][1].split()[-1] + data.split()[0]
                        self.annotate_label_idx(error_text, correct_text, error, correct)
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
                                        self.annotate_label_idx(error_text, correct_text, error, correct)
                                    elif diff[idx - 1][1][-1] == ' ':
                                        error_text = diff[idx - 2][1].split()[-1] + diff[idx - 1][1] + data.split()[0]
                                        correct_text = diff[idx - 2][1].split()[-1] + data.split()[0]
                                        self.annotate_label_idx(error_text, correct_text, error, correct)
                                    # 추가 오류
                                    # correct: "예전에 있었다." -> error: "예전에 있었었다."
                                    else:
                                        error_text = diff[idx - 2][1].split()[-1] + diff[idx - 1][1] + data.split()[0]
                                        correct_text = diff[idx - 2][1].split()[-1] + data.split()[0]
                                        self.annotate_label_idx(error_text, correct_text, error, correct)
                                else:
                                    # 추가 오류
                                    # correct: "나는 안양에서 일한다." -> error: "나는 안양에서 서 일한다."
                                    if data[0] == ' ' or diff[idx - 1][1][-1] == ' ':
                                        error_text = diff[idx - 2][1].split()[-1] + ' ' + diff[idx - 1][1]
                                        correct_text = diff[idx - 2][1].split()[-1] + ' '
                                        self.annotate_label_idx(error_text, correct_text, error, correct)
                                    # 추가 오류
                                    # correct: "나는 김포시에서 일했다." -> error: "나는 ㄱ김포시에서 일했다."
                                    else:
                                        error_text = diff[idx - 1][1] + data.split()[0]
                                        correct_text = data.split()[0]
                                        self.annotate_label_idx(error_text, correct_text, error, correct)
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
                                            self.annotate_label_idx(error_text, correct_text, error, correct)

                                        else:
                                            # 교체 오류
                                            # correct: "국내에서 온천(?) 갈까 생각 중이야." -> error: "국내에서 온천?갈까 생각중이야"
                                            if diff[idx - 2][1][-1] == ' ':
                                                if data[0] == ' ':
                                                    error_text = diff[idx - 3][1].split()[-1] + diff[idx - 1][1]
                                                    correct_text = diff[idx - 3][1].split()[-1] + diff[idx - 2][1]
                                                    self.annotate_label_idx(error_text, correct_text, error, correct)
                                                else:
                                                    error_text = diff[idx - 3][1].split()[-1] + diff[idx - 1][1] + \
                                                                 data.split()[0]
                                                    correct_text = diff[idx - 3][1].split()[-1] + diff[idx - 2][1] + \
                                                                   data.split()[0]
                                                    self.annotate_label_idx(error_text, correct_text, error, correct)
                                            # 교체 오류
                                            # correct: "의자가 부셔졌다." -> error: "의자가 부셨다."
                                            else:
                                                if diff[idx - 2][1][0] == ' ':
                                                    error_text = diff[idx - 1][1] + data.split()[0]
                                                    correct_text = diff[idx - 2][1] + data.split()[0]
                                                    self.annotate_label_idx(error_text, correct_text, error, correct)
                                                else:
                                                    error_text = diff[idx - 3][1].split()[-1] + diff[idx - 1][
                                                        1] + data.split()[0]
                                                    correct_text = diff[idx - 3][1].split()[-1] + diff[idx - 2][
                                                        1] + data.split()[0]
                                                    self.annotate_label_idx(error_text, correct_text, error,
                                                                            correct)
                                    else:
                                        # 교체 오류
                                        # correct: "이제 곧 들어가야 해" -> error: "이제 곶 들어가야 해"
                                        if data[0] == ' ':
                                            error_text = diff[idx - 1][1] + ' ' + data.split()[0]
                                            correct_text = diff[idx - 2][1] + ' ' + data.split()[0]
                                            self.annotate_label_idx(error_text, correct_text, error, correct)
                                        # 교체 오류
                                        # correct: "이 길로 곧장 가야한다." -> error: "이 길로 곶장 가야한다."
                                        else:
                                            if diff[idx - 3][1] == ' ':
                                                error_text = diff[idx - 3][1] + diff[idx - 1][1] + data.split()[0]
                                                correct_text = diff[idx - 2][1] + data.split()[0]
                                                self.annotate_label_idx(error_text, correct_text, error, correct)
                                            elif diff[idx - 3][1][-1] == ' ':
                                                error_text = diff[idx - 3][1].split()[-1] + ' ' + diff[idx - 1][1] + \
                                                             data.split()[0]
                                                correct_text = diff[idx - 3][1].split()[-1] + ' ' + diff[idx - 2][1] + \
                                                               data.split()[0]
                                                self.annotate_label_idx(error_text, correct_text, error, correct)
                                            else:
                                                error_text = diff[idx - 3][1].split()[-1] + diff[idx - 1][1] + \
                                                             data.split()[0]
                                                correct_text = diff[idx - 3][1].split()[-1] + diff[idx - 2][1] + \
                                                               data.split()[0]
                                                self.annotate_label_idx(error_text, correct_text, error, correct)
                                else:
                                    # 교체 오류
                                    # correct: "곧 경기가 시작한다." -> error: "곶 경기가 시작한다."
                                    if data[0] == ' ':
                                        error_text = diff[idx - 1][1] + ' ' + data.split()[0]
                                        correct_text = diff[idx - 2][1] + ' ' + data.split()[0]
                                        self.annotate_label_idx(error_text, correct_text, error, correct)
                                    else:
                                        error_text = diff[idx - 1][1] + data.split()[0]
                                        correct_text = diff[idx - 2][1] + data.split()[0]
                                        self.annotate_label_idx(error_text, correct_text, error, correct)

                        else:
                            # 추가 오류
                            # correct: " 1. 연구 목적" -> error: ". 1. 연구 목적"
                            if data[0] == ' ':
                                error_text = diff[idx - 1][1] + ' ' + data.split()[0]
                                correct_text = ' ' + data.split()[0]
                                self.annotate_label_idx(error_text, correct_text, error, correct)
                            # 추가 오류
                            # correct: "지금 이 기세라면" -> error: "ㅈ 지금 이 기세라면"
                            elif diff[idx - 1][1][-1] == ' ':
                                error_text = diff[idx - 1][1] + data.split()[0]
                                correct_text = data.split()[0]
                                self.annotate_label_idx(error_text, correct_text, error, correct)
                            # 추가 오류
                            # correct: "지금 이 기세라면" -> "ㅈ지금 이 기세라면"
                            else:
                                error_text = diff[idx - 1][1] + data.split()[0]
                                correct_text = data.split()[0]
                                self.annotate_label_idx(error_text, correct_text, error, correct)



        tmp_error_label = self.error_label.copy()
        pattern = re.compile(r'E+')
        e_flag = True
        tag_len = 0
        while e_flag:
            s_i, e_i = pattern.search(''.join(tmp_error_label)).span()
            e_word = error[s_i:e_i]
            self.annotation[s_i+tag_len:e_i+tag_len] = f'<S{self.n}>{e_word}</S{self.n}>'
            tag_len += len(f'<S{self.n}>') + len(f'</S{self.n}>')
            tmp_error_label[s_i:e_i] = ['O'] * (e_i-s_i)
            if e_i - s_i == 1:
                self.error_label[s_i:e_i] = ['B-E']
            elif e_i - s_i == 2:
                self.error_label[s_i:s_i + 1] = ['B-E']
                self.error_label[s_i + 1:e_i] = ['E-E']
            else:
                self.error_label[s_i:s_i + 1] = ['B-E']
                self.error_label[s_i + 1:e_i - 1] = ['I-E'] * (e_i - s_i - 2)
                self.error_label[e_i - 1:e_i] = ['E-E']

            self.n += 1
            if 'E' not in tmp_error_label:
                e_flag = False
                self.n = 1

        if len(error) != len(self.error_label):
            return [], []

        self.annotation = ''.join(self.annotation)

        pattern = re.compile(r'C+')
        c_flag = True
        while c_flag:
            s_i, e_i = pattern.search(''.join(self.correct_label)).span()
            s_word = correct[s_i:e_i]
            self.correction.append(f'{s_word}')
            self.correct_label[s_i:e_i] = ['O'] * (e_i - s_i)
            self.n += 1
            if 'C' not in self.correct_label:
                c_flag = False
                self.n = 1

        self.error_label = ' '.join(self.error_label)
        self.correction = '<sep>'.join(self.correction)
        return error, self.error_label


if __name__ == '__main__':
    input_path = 'original_dataset/sample.txt'
    with open(input_path, 'r', encoding='utf-8') as rf:
        lines = rf.readlines()

    annotation_tool = ErrorAnnotation()

    examples = []
    for line in tqdm(lines):
        line = line.strip()

        if '\t' in line:
            e, c = line.split('\t')
            e = e.replace(' ', ' ')
            c = c.replace(' ', ' ')
            e = e.replace('​', ' ')
            c = c.replace('​', ' ')

            flag = True
            while flag:
                e = e.replace('  ', ' ')
                c = c.replace('  ', ' ')
                if '  ' not in e and '  ' not in c:
                    flag = False

            e = e.strip(' ')
            c = c.strip(' ')

            if c in except_lst:
                continue

            if e == '' or c == '':
                continue

            if e == '``':
                continue

            if e == c:
                label = ' '.join(['O'] * len(e))
                examples.append({'error': e, 'tag': label})
                continue

            error, label  = annotation_tool.annotate_error(c, e)
            if not error:
                continue

            examples.append({'error': error, 'tag': label})

    with open('output_data/output_ged_sample.jsonl', 'w', encoding='utf-8') as wf:
        for example in examples:
            json.dump(example, wf, ensure_ascii=False)
            wf.write('\n')
