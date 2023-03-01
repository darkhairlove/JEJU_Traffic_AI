# 제주도 도로 교통량 예측 AI 경진대회

---
## 대회 설명
제주도 도로 교통량 예측 AI 알고리즘 개발  
제주도의 교통 정보로부터 도로 교통량 회귀 예측



## Structure

```
Folder/
|- EDA/          # EDA (ipynb)
|- model/        # final model (py)
|- reference/    # paper (pdf)
```

## Dataset
**Data Source**

[Train Test Dateset](https://dacon.io/competitions/official/235985/overview/description)

[날씨](https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36&tabNo=2#)  
```
Dataset Info.
train.csv : 2022년 8월 이전 데이터만 존재하며 날짜, 시간,
            교통 및 도로구간 등의 정보와 도로의 차량 평균 속도(target)정보 포함
    
test.csv : 2022년 8월 데이터만 존재하며 날짜, 시간, 교통 및
           도로구간 등의 정보 포함

weather.csv : 제주도 날씨 데이터
              일시, 기온, 강수량, 풍속, 안개 계속 시간 등
```

## Modeling

- 0.8*XGBoost + 0.15*LightGBM + 0.05*ExtraTree
- 각 모델을 FLAML을 활용해 하이퍼 파라미터 튜닝

## **Member**
- [김경민](https://github.com/wonderkyeom)
- [김태종](https://github.com/xowhddk123)
- [박수진](https://github.com/darkhairlove)
- [조근혜](https://github.com/GH-Jo)

## Result
- 평가 지표 : **MAE**
- **Private score** 19등 3.10782
- **Public score** 17등 3.09801 

## Comments
[느낀점](https://www.notion.so/40c65577c29b46b585cc8712a5b060d9?pvs=4#a401159915684fbdab79d7c8d2cb680b)