# Matplotlib 기초 튜토리얼

## 📊 개요
Python의 대표적인 데이터 시각화 라이브러리인 Matplotlib의 기본 사용법을 정리한 문서입니다.

## 📚 목차
1. [Line Plot (선 그래프)](#1-line-plot-선-그래프)
2. [Bar Graph (막대 그래프)](#2-bar-graph-막대-그래프)

---

## 1. Line Plot (선 그래프)

### 기본 코드
```python
import numpy as np
import matplotlib.pyplot as plt

# 그래프 제목 설정
plt.title("Graph tiltle", fontsize=20, fontweight='bold', 
          loc='right', color='blue')

# 선 그래프 그리기
plt.plot([1,2,3], [4,5,6], linewidth=1, c='g', linestyle=':')
plt.plot([1,2,3], [1,4,9])

# 그리드 추가
plt.grid(linestyle=':')

# 축 라벨 설정
plt.xlabel('Sequence', fontsize=12, fontweight='bold')
plt.ylabel('Time(secs)', fontsize=12, fontweight='bold')

# 범례 추가
plt.legend(['Mouse', 'Cat'])

# 축 범위 설정
plt.xlim([0, 4])      # X축의 범위: [xmin, xmax]
plt.ylim([0, 10])     # Y축의 범위: [ymin, ymax]

plt.show()
```

### 주요 파라미터 설명

#### 📌 제목 설정 (`plt.title()`)
- `fontsize`: 글자 크기
- `fontweight`: 글자 굵기 ('normal', 'bold' 등)
- `loc`: 제목 위치 ('left', 'center', 'right')
- `color`: 글자 색상

#### 📌 선 스타일 (`plt.plot()`)
- `linewidth`: 선 두께
- `c` 또는 `color`: 선 색상
- `linestyle`: 선 스타일
  - `'-'`: 실선 (기본값)
  - `'--'`: 대시선
  - `'-.'`: 대시-점선
  - `':'`: 점선

#### 📌 그리드 설정 (`plt.grid()`)
- `linestyle`: 그리드 선 스타일

#### 📌 축 설정
- `plt.xlabel()`, `plt.ylabel()`: 축 라벨 설정
- `plt.xlim()`, `plt.ylim()`: 축 범위 설정
- `plt.legend()`: 범례 추가

### 다양한 선 그래프 예제

```python
# 기본 선 그래프
plt.plot([2,3,4,5], linewidth=1, c='b')

# X, Y 좌표를 지정한 선 그래프
plt.plot([1,2,3], [4,5,6], linewidth=1, c='g')

# 다양한 선 스타일
plt.plot([1,2,3], [4,5,6], linewidth=1, c='g', linestyle='--')  # 대시선
plt.plot([1,2,3], [4,5,6], linewidth=1, c='g', linestyle='-.')  # 대시-점선
plt.plot([1,2,3], [4,5,6], linewidth=1, c='g', linestyle=':')   # 점선
```

---

## 2. Bar Graph (막대 그래프)

### 기본 코드
```python
import numpy as np
import matplotlib.pyplot as plt

# 데이터 준비
y = [5, 3, 7, 10, 9, 5, 3.5, 8]
x = range(len(y))

# 그래프 제목
plt.title("Bar Graph", fontsize=20, fontweight='bold', 
          loc='center', color='blue')

# 조건에 따른 색상 설정
colors = ['red' if val > 7 else 'blue' for val in y]

# 막대 그래프 그리기
plt.bar(x, y, width=0.4, color=colors)

# 축 라벨
plt.xlabel('Sequence', fontsize=12, fontweight='bold')
plt.ylabel('Time(secs)', fontsize=12, fontweight='bold')

plt.show()
```

### 주요 파라미터 설명

#### 📌 막대 그래프 설정 (`plt.bar()`)
- `x`: X축 위치 값
- `y`: 막대의 높이 값
- `width`: 막대의 너비
- `color`: 막대 색상 (단일 색상 또는 리스트)

#### 📌 조건부 색상 적용
```python
# 값이 7보다 크면 빨간색, 아니면 파란색
colors = ['red' if val > 7 else 'blue' for val in y]
```

이 방식을 통해 데이터 값에 따라 동적으로 색상을 변경할 수 있습니다.

---

## 💡 활용 팁

### 1. 여러 그래프를 한 화면에 표시
```python
# 여러 선을 한 그래프에 표시
plt.plot([1,2,3], [4,5,6], label='Line 1')
plt.plot([1,2,3], [1,4,9], label='Line 2')
plt.legend()
```

### 2. 그래프 저장
```python
plt.savefig('graph.png', dpi=300, bbox_inches='tight')
```

### 3. 한글 폰트 설정 (한글 사용 시)
```python
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', family='Malgun Gothic')  # Windows
# rc('font', family='AppleGothic')  # Mac
plt.rcParams['axes.unicode_minus'] = False
```

---

## 📚 참고 자료
- [Matplotlib 공식 문서](https://matplotlib.org/stable/contents.html)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)

---

## 📝 License
이 문서는 학습 목적으로 자유롭게 사용할 수 있습니다.