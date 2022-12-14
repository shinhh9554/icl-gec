# Pipeline-based Korean Grammatical Error Correction System

딥러닝 기반 파이프라인 한국어 문법 오류 교정 통합 시스템

## 1. 개요

문법 오류 교정(Grammar Error Correction, GEC)은 기존의 규칙 및 통계기반 방법으로 연구가 되었으나 시간과 비용의 한계로 성능 향상에 한계가 존재했다.
이에 최근 연구는 딥러닝 기술을 적용하여 기존 방법의 한계를 보완하였다. 딥러닝 기술을 적용하여 기존 방법의 한계를 보완하였다. 
하지만 기존 딥러닝 기반 문법 오류 교정 모델은 문법 오류 감지와 교정이 동시에 수행되어 문법 오류가 없는 문장을 똑같이 복사하는 비효일성을 발생시키고 수정된 결과에 대해서 명확한 설명을 도출하지 못하는 한계가 존재한다.
따라서 본 연구에서 효과적으로 문법 오류를 교정하고 이용자들에게 결과에 대해 설명할 수 있는 딥러닝 기반 문법 오류 교정 시스템을 제안하였습니다.
이와 관련된 논문은 [여기서](http://riss4u.net/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=3b1c608011255b07ffe0bdc3ef48d419&keyword=%EC%8B%A0%ED%98%84%ED%98%B8%20%EC%B5%9C%EC%84%B1%ED%95%84) 확인하실 수 있습니다.

## 2. 모델 구조

파이프라인 문법 오류 교정 시스템은 문법 오류 교정 과정을 문법 오류 감지, 문법 오류 수정, 문법 오류 유형 분류로 나누어 3단계로 진행함.

<img width="716" alt="시스템 구조도" src="https://user-images.githubusercontent.com/57481142/200285534-7df892ad-50f8-4364-87c3-0e916ea97f6a.png">

### 2-1. 문법 오류 감지 모델

* 문법 오류 감지 모델은 주어진 문장에서 문법 오류의 존재와 그 위치를 감지하는 역할을 수행함.
* [monologg/KoCharELECTRA](https://github.com/monologg/KoCharELECTRA) 을 파인튜닝하여 모델을 구축함.
* 학습 데이터 샘플
```
{"error": "최소 3달전에 예약하는게 안전빠", "tag": "O O O B-E I-E I-E E-E O B-E I-E I-E I-E E-E O B-E I-E E-E"}
{"error": "ㅋㅋ아 그러시군요! ㅎㅎ", "tag": "B-E I-E I-E E-E O O O O O O O O O"}
{"error": "아 그거 괜찮네요", "tag": "B-E E-E O O O B-E I-E I-E E-E"}
```

### 2-2. 문법 오류 교정 모델

* 문법 오류 교정 모델은 주어진 문법 오류 문장과 위치 정보를 통해 문법 오류 문구를 순차적으로 교정하는 역할을 수행함.
* [SKT-AI/KoBART](https://github.com/SKT-AI/KoBART)을 기반으로 아래와 같이 문법적으로 오류가 난 영역을 특수 토큰으로 감싼 후 순차적으로 문법 오류를 교정
* 학습 데이터 샘플
```
{"error": "간편하겐 <unused0>정말강추인데<unused1>", "correct": "정말 강추인데"}
{"error": "그때그때 그라인더에 갈면 향이 정말 <unused0>좋아요<unused1>", "correct": "좋아요."}
{"error": "커피 <unused0>안마시지만<unused1> 이뻐서 일리는 사고싶어요ㅋㅋ", "correct": "안 마시지만"}
{"error": "커피 안 마시지만 이뻐서 일리는 <unused0>사고싶어요ㅋㅋ<unused1>", "correct": "사고 싶어요. ㅋㅋ"}
```


### 2-3. 문법 오류 유형 분류 모델

* 문법 오류 교정 모델은 주어진 문법 오류 문장과 위치 정보를 통해 문법 오류 문구를 순차적으로 교정하는 역할을 수행함.
* [monologg/KoCharELECTRA](https://github.com/monologg/KoCharELECTRA) 을 파인튜닝하여 모델을 구축함.
* 학습 데이터 샘플

| 오류 문구       | 교정 문구        | 오류 유형   |
|-------------|--------------|---------|
| 줬는데 리쁘드랑    | 줬는데 이쁘더라     | 5       |
| 줬는데 리쁘드랑~~~ | 줬는데 이쁘더라~~~  | 5       |
| 줬는데 지가봐놓고   | 줬는데 자기가 봐 놓고 | 5       |
| 줬다고 가는      | 줬다고 걔는       | 4       |
| 줬다고 가는      | 줬다고? 걔는      | 4       |
| 쥐똥만큼밖에      | 쥐똥만큼 밖에      | 1       |
| 쥐똥만하게       | 쥐똥만 하게       | 1       |

<br/>

* 오류 유형 분류 표

<img width="453" alt="스크린샷 2022-12-14 오후 4 35 21" src="https://user-images.githubusercontent.com/57481142/207533835-e8141bfb-6c53-48b7-ab4a-60237bdeae68.png">


## 3. How To Run

### 3-1. 문법 오류 감지 모델
👉 ged directory 로 이동
```
cd ged
```

👉 모든 GPU를 활용하여 모델 학습을 수행
```
python train.py \
  --model_name_or_path monologg/kocharelectra-base-discriminator \
  --train_file data/ged_train.jsonl \
  --validation_file data/ged_valid.jsonl \
  --max_seq_length 128 \
  --output_dir output \
  --num_train_epochs 5.0 \
```

👉 특정 GPU를 활용하여 모델 학습을 수행
```
CUDA_VISIBLE_DEVICES=0,2 python train.py \
  --model_name_or_path monologg/kocharelectra-base-discriminator \
  --train_file data/ged_train.jsonl \
  --validation_file data/ged_valid.jsonl \
  --max_seq_length 128 \
  --output_dir output \
  --num_train_epochs 5.0 \
```

### 3-2. 문법 오류 교정 모델
👉 gec directory 로 이동
```
cd gec
```

👉 모든 GPU를 활용하여 모델 학습을 수행
```
python train.py \
  --model_name_or_path gogamza/kobart-base-v2 \
  --train_file data/gec_train.jsonl \
  --validation_file data/gec_valid.jsonl \
  --max_seq_length 128 \
  --output_dir output \
  --num_train_epochs 5.0 \
```

👉 특정 GPU를 활용하여 모델 학습을 수행
```
CUDA_VISIBLE_DEVICES=0,2 python train.py \
  --model_name_or_path gogamza/kobart-base-v2 \
  --train_file data/gec_train.jsonl \
  --validation_file data/gec_valid.jsonl \
  --max_seq_length 128 \
  --output_dir output \
  --num_train_epochs 5.0 \
```

### 3-3. 문법 오류 유형 분류 모델
👉 cls directory 로 이동
```
cd cls
```

👉 모델 학습 수행
```
python main.py \
  --model_type koelectra \
  --data_path data/new_noise_sample.txt \
  --model_dir "./models" \
  --num_train_epochs 5 \
  --batch_size 180 \
  --max_seq_length 128 \
```


### 3-4. 데이터 전처리
👉 문법 오류 감지 모델 데이터의 경우 아래 경로에 있는 파일을 실행하여 데이터 전처리를 수행한다.
```
cd etc/preprocess
```
```
python annotate_data_ged.py
```
👉 문법 오류 교정 모델 데이터의 경우 아래 파일 경로에 있는 파일을 실행하여 데이터 전처리를 수행한다.
```
cd etc/preprocess
```
```
python annotate_data_gec.py
```

### 3-5. 학습 데이터 증강 (노이즈 데이터 생성)
👉 문법 오류가 없는 문장에 새로운 노이즈를 생성하려면 아래 경로에 있는 파일을 실행하여 노이즈 데이터를 생성한다.
```
cd etc/noise_data
```
```
python apply_noise.py
```


# 관련 내용 작성 중 입니다.!