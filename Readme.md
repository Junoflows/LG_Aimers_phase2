# LG Aimers phase2

### MQL 데이터를 활용하여 영업 기회 전환 고객을 선별하기 위한 AI 모델 개발
팀원 : [송준호](https://github.com/Junoflows), [한지성](https://github.com/jisung99), [황성주](https://github.com/svng-zu)

### 최종 결과
+ Public Score : 0.75611
+ Final Score : 0.76485

#### 844팀 중 63위 (30위 팀 Final Score : 0.78086)


## 1. 개요
#### [설명]
+ MQL데이터를 활용하여 영업 기회 전환 고객을 선별하기 위한 AI모델 개발합니다.
+ 온라인 해커톤에서 교육생들의 문제 해결 능력을 검증하여 오프라인 해커톤에 진출할 약 100명을 선별하기 위한 과정입니다.

#### [주최 / 주관]
+ 주최 : LG AI Research
+ 주관 : 엘리스그룹
+ 참여 : 한경닷컴

#### [리더보드]
+ 평가 산식 : F1 score
+ Public score : 전체 테스트 데이터 샘플 중 사전 샘플링된 50%로 계산
+ Private score : Public score 계산에 포함되지 않은 나머지 50%의 테스트 데이터로 계산
  
## 2. EDA
### 전체 데이터 확인
<img src="https://github.com/svng-zu/LG-AIMERS/assets/70852514/9bae032e-b21f-42cf-bb43-fda2c30f461e" alt="data1" width="30%" height="30%">
<img src="https://github.com/svng-zu/LG-AIMERS/assets/70852514/06ec7602-f1b7-4223-9e87-240d15f620f9" alt="data1" width="30%" height="30%"> <br/>

+ 결측치가 많은 데이터 임을 알 수 있음

<img src="https://github.com/svng-zu/LG-AIMERS/assets/70852514/fc7496a8-71a4-4e15-9745-3ab54bb6fa59" alt="data1" width="50%" height="50%">

+ 타겟 변수 is_converted 의 True, False 비율이 약 11: 1로 불균형이 있음을 알 수 있음
+ 클래스 불균형 데이터는 모델 학습 시 과적합의 위험이 큼
  + 언더샘플링으로 해결

<br/>

### 범주형 변수
<img src="https://github.com/svng-zu/LG-AIMERS/assets/70852514/c3bd18ae-8371-4e30-8cad-2c185750fd4c" alt="data" width=30% height=30%>
<img src="https://github.com/svng-zu/LG-AIMERS/assets/70852514/ee3fd6df-0429-46b3-927c-ac8293a9e9bf" alt="data" width=30% height=30%>
<img src="https://github.com/svng-zu/LG-AIMERS/assets/70852514/fdb061c3-9a6c-4c4c-aa0f-f3c916246ef3" alt="data" width=30% height=30%>

+ 대부분의 변수에서 희소 범주의 개수가 많음
+ 희소 범주들은 모델 학습 시 트리의 깊이가 깊어지고 과적합으로 모델 성능 저하시킴
  + 희소 범주를 '기타' 범주로 처리

### 수치형 변수
<img src="https://github.com/svng-zu/LG-AIMERS/assets/70852514/ccff87f9-e838-4de0-aec4-47fe1caab7e8" alt="data" width=50% height=50%>

<br/>

### 상관계수
<img src="https://github.com/svng-zu/LG-AIMERS/assets/70852514/5d40481f-2a2b-4191-be82-bd957a022311" alt="data" width=50% height=50%>

+ 변수간의 상관성이 적음을 알 수 있음
  
<br/>

## 3. 데이터 전처리
### 컬럼 삭제
+ 중복 변수 제거 : customer_country.1
+ 결측치가 과반인 변수는 제거 : product_subcategory, product_modelname, business_area, business_subarea
+ 변수 중요도가 매우 낮은 변수 제거 : id_strategic_ver, it_strategic_ver, idit_strategic_ver, ver_cus, ver_pro

<img src="https://github.com/svng-zu/LG-AIMERS/assets/70852514/0b20e0d1-c783-4a7f-a6d9-bcb66c09792e" alt="data" width=70% height=70%>

### 같은 의미의 다른 데이터는 같은 범주로 처리
+ etc, other, others $\rightarrow$ etc
+ end-customer, end customer, end-user $\rightarrow$ end_user

<br/>

### 개수가 1개인 범주들을 기타 처리
+ 결측치와는 다르게 처리

<br/>

### 결측치 처리
+ 수치형 데이터는 0 대체해도 무방 했음
+ 범주형은 None이라는 문자열로 범주처럼 처리
<br/>

## 4. 모델링

### 모델 선택
#### autoML - pycaret 사용

__pycaret__ <br/>
ML workflow을 자동화 하는 opensource library로 여러 머신러닝 task에서 사용하는 모델들을 하나의 환경에서 비교하고 튜닝하는 등 간단한 코드를 통해 편리하게 사용할 수 있도록 자동화환 라이브러리

#### autoML 실행결과
<img src="https://github.com/svng-zu/LG-AIMERS/assets/70852514/1571c80d-74b2-40ce-bf06-91e4caea475e" alt="data" width=70% height=70%>

+ 각 모델에 대해서 어떤 모델을, 몇 개를 조합할 것인지에 대한 실험이 필요

<br/>

## 5. 과적합 핸들링

### 1. 언더샘플링
+ 타겟 변수의 True와 False 값의 비율이 약 11:1로 클래스 불균형이 심한 상태임
+ 이를 그대로 학습하게 되면 False 클래스에 편향된 모델이 되기 때문에 오버 샘플링 / 언더 샘플링을 진행
+ 실험 결과 언더 샘플링의 F1-score가 더 높아 언더 샘플링을 진행

정보 손실의 위험 $\rightarrow$ 앙상블 + 보팅으로 해결

<img src="https://github.com/svng-zu/LG-AIMERS/assets/70852514/cec9466f-2b4e-4ecc-9aae-41c4f86337ef" alt="data" width=50% height=50% left=0>

+ public score 0.2.. 에서 0.6 대로 상승

<br/>

### 2. 앙상블
+ 여러개의 예측 모델을 결합하여 과적합을 줄이고 모델을 일반화하는 방법
+ 앞서 고른 상위 5개 모델을 앙상블하여 모델 일반화 진행 함.

<br/>

### 3. 모델 학습 시 편향되어 학습되는 요인 찾기
+ train 데이터에서 customer_idx = 25096 의 경우 영업 횟수 2421 모두 성공한 것으로 관측됨. 
+ train 데이터의 True 개수가 4850개 임을 생각하면 위 idx에 편향되어 학습된다고 판단함
+ pubilc score 0.7대로 상승

<br/>

### 4. Voting
+ 언더 샘플링 시 정보손실의 문제가 있음.
+ False 데이터 54449 개를 랜덤 셔플 후, 모두 20등분하고 True와 합쳐 클래스 비율이 1:1인 데이터셋 20개를 생성.
+ 각 셔플의 모델에서의 결과를 확률로 받은 후 0, 1 클래스의 확률을 평균을 내어 최종 결과로 생성 (Soft voting)
$\rightarrow$ public score 0.02 정도 상승을 보임

<br/>

## 6. AB test
![image](https://github.com/Junoflows/LG_Aimers_phase2/assets/108385417/2717cbda-04ff-43da-87d6-7e2d8b8ca309)

+ 모델 학습은 GridSearch를 이용

<br/>

## 7. 결과 및 리뷰

### 모델 선택
앞서 선택한 5개 모델 중 5개, 3개, 1개로 나누어 앙상블 후 가장 public score가 높은 모델 선택
+ 'xgb' 1개 사용시 가장 성능이 높음.

### 리뷰
+ 실제 현업에서 사용하는 데이터는 전처리에 많은 시간을 쏟아야 한다는 것을 느낌
+ 과적합을 해결하기 위해 많은 고민을 했고 그 과정에서 데이터와 모델에 대한 이해를 키울 수 있었음
+ AutoML 의 pycaret 을 일찍 적용했더라면 실험 과정의 튜닝 시간을 효율적으로 사용했을 것 같음
+ 임의로 설정한 값들에 대해 정확한 튜닝을 하지 못하였던 것이 아쉬움(시간 및 제출 횟수 부족)

