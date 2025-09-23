# NumPy 학습 정리 📚

## 목차
1. [NumPy 기본 개념](#1-numpy-기본-개념)
2. [배열 생성 방법](#2-배열-생성-방법)
3. [배열 속성과 정보](#3-배열-속성과-정보)
4. [배열 인덱싱과 슬라이싱](#4-배열-인덱싱과-슬라이싱)
5. [배열 연산](#5-배열-연산)
6. [통계 연산과 집계 함수](#6-통계-연산과-집계-함수)
7. [Axis 개념 완전 정복](#7-axis-개념-완전-정복)
8. [실전 예제](#8-실전-예제)
9. [요약 및 핵심 포인트](#9-요약-및-핵심-포인트)

---

## 1. NumPy 기본 개념

### 1.1 NumPy란?
- **Numerical Python**의 줄임말
- 파이썬에서 **수치 계산**을 위한 핵심 라이브러리
- **다차원 배열 객체(ndarray)**와 이를 다루는 도구들을 제공
- 과학 계산, 데이터 분석, 머신러닝의 기초

### 1.2 NumPy Array vs Python List

| 특성 | Python List | NumPy Array |
|------|-------------|-------------|
| **데이터 타입** | 혼합 가능 | 단일 타입 |
| **메모리 사용** | 비효율적 | 효율적 |
| **연산 속도** | 느림 | 빠름 (C로 구현) |
| **수학 연산** | 지원 안함 | 벡터화 연산 지원 |
| **크기** | 가변 | 고정 (생성 후) |

### 1.3 기본 사용법

```python
import numpy as np  # 관례적으로 np로 alias

# Python 리스트를 NumPy 배열로 변환
python_list = [[1, 2, 3], [4, 5, 6]]
numpy_array = np.array(python_list)

print(f"Python List Type: {type(python_list)}")  # <class 'list'>
print(f"NumPy Array Type: {type(numpy_array)}")  # <class 'numpy.ndarray'>
```

---

## 2. 배열 생성 방법

### 2.1 리스트에서 배열 생성

```python
import numpy as np

# 1차원 배열
arr_1d = np.array([1, 2, 3, 4, 5])
print(f"1D Array: {arr_1d}")

# 2차원 배열 (행렬)
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"2D Array:\n{arr_2d}")

# 3차원 배열
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"3D Array:\n{arr_3d}")
```

### 2.2 특수 배열 생성 함수

#### 2.2.1 영배열 (zeros)
```python
# 모든 요소가 0인 배열
zeros_1d = np.zeros(5)          # [0. 0. 0. 0. 0.]
zeros_2d = np.zeros((2, 3))     # 2행 3열의 0 배열
zeros_3d = np.zeros((2, 2, 2))  # 2x2x2의 3차원 0 배열

print(f"1D Zeros: {zeros_1d}")
print(f"2D Zeros:\n{zeros_2d}")
```

#### 2.2.2 일배열 (ones)
```python
# 모든 요소가 1인 배열
ones_1d = np.ones(4)            # [1. 1. 1. 1.]
ones_2d = np.ones((3, 2))       # 3행 2열의 1 배열

print(f"1D Ones: {ones_1d}")
print(f"2D Ones:\n{ones_2d}")
```

#### 2.2.3 특정 값으로 채우기 (full)
```python
# 모든 요소를 특정 값으로 채우기
full_array = np.full((2, 3), 7)  # 2행 3열을 7로 채움
print(f"Full Array:\n{full_array}")

# 다른 배열과 같은 shape로 특정 값 채우기
template = np.array([[1, 2], [3, 4]])
full_like = np.full_like(template, 9)
print(f"Full Like:\n{full_like}")
```

#### 2.2.4 기타 유용한 생성 함수
```python
# 단위 행렬 (identity matrix)
identity = np.eye(3)  # 3x3 단위 행렬
print(f"Identity Matrix:\n{identity}")

# 범위 배열
range_array = np.arange(0, 10, 2)  # 0부터 10까지 2씩 증가
print(f"Range Array: {range_array}")

# 균등 분할
linspace_array = np.linspace(0, 1, 5)  # 0부터 1까지 5개로 균등 분할
print(f"Linspace Array: {linspace_array}")
```

---

## 3. 배열 속성과 정보

### 3.1 주요 속성들

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(f"배열:\n{arr}")
print(f"차원 수 (ndim): {arr.ndim}")      # 2 (2차원)
print(f"형태 (shape): {arr.shape}")       # (2, 3) - 2행 3열
print(f"크기 (size): {arr.size}")         # 6 (총 요소 개수)
print(f"데이터 타입 (dtype): {arr.dtype}") # int64 (정수형)
print(f"각 요소 크기 (itemsize): {arr.itemsize}") # 8 bytes
print(f"전체 메모리 사용량: {arr.nbytes}") # 48 bytes
```

### 3.2 데이터 타입 (dtype)

```python
# 정수형
int_array = np.array([1, 2, 3], dtype=np.int32)
print(f"Integer Array: {int_array}, dtype: {int_array.dtype}")

# 실수형
float_array = np.array([1.1, 2.2, 3.3], dtype=np.float64)
print(f"Float Array: {float_array}, dtype: {float_array.dtype}")

# 불리언
bool_array = np.array([True, False, True], dtype=np.bool_)
print(f"Boolean Array: {bool_array}, dtype: {bool_array.dtype}")

# 타입 변환
converted = int_array.astype(np.float64)
print(f"Converted Array: {converted}, dtype: {converted.dtype}")
```

---

## 4. 배열 인덱싱과 슬라이싱

### 4.1 기본 인덱싱

```python
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Original Array:\n{arr_2d}")

# 개별 요소 접근
print(f"첫 번째 행, 첫 번째 열: {arr_2d[0, 0]}")  # 1
print(f"두 번째 행, 세 번째 열: {arr_2d[1, 2]}")  # 6
print(f"마지막 행, 마지막 열: {arr_2d[-1, -1]}")  # 9

# 전체 행 또는 열 접근
print(f"첫 번째 행 전체: {arr_2d[0, :]}")        # [1 2 3]
print(f"첫 번째 열 전체: {arr_2d[:, 0]}")        # [1 4 7]
```

### 4.2 슬라이싱 (Slicing)

#### 4.2.1 기본 슬라이싱
```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 부분 배열 추출
sub_array1 = arr[0:2, 0:2]  # 0~1행, 0~1열
print(f"Sub Array 1:\n{sub_array1}")
# Output:
# [[1 2]
#  [4 5]]

sub_array2 = arr[1:3, 1:3]  # 1~2행, 1~2열
print(f"Sub Array 2:\n{sub_array2}")
# Output:
# [[5 6]
#  [8 9]]
```

#### 4.2.2 고급 슬라이싱
```python
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# 끝에서부터 슬라이싱
print(f"마지막 2행, 마지막 2열:\n{arr[-2:, -2:]}")

# 스텝 사용
print(f"모든 행, 격칸 열: {arr[:, ::2]}")  # 0, 2번째 열만

# 역순
print(f"역순 배열:\n{arr[::-1, ::-1]}")    # 행과 열 모두 역순
```

### 4.3 불리언 인덱싱

```python
arr = np.array([1, 2, 3, 4, 5, 6])

# 조건에 맞는 요소만 선택
condition = arr > 3
print(f"조건 (>3): {condition}")           # [False False False True True True]
print(f"조건에 맞는 요소들: {arr[condition]}")  # [4 5 6]

# 직접 조건 사용
print(f"짝수만 선택: {arr[arr % 2 == 0]}")   # [2 4 6]

# 2차원 배열에서 조건 사용
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"3보다 큰 요소들: {arr_2d[arr_2d > 3]}")  # [4 5 6]
```

---

## 5. 배열 연산

### 5.1 요소별 연산 (Element-wise Operations)

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(f"a = {a}")
print(f"b = {b}")

# 기본 산술 연산
print(f"덧셈: a + b = {a + b}")           # [5 7 9]
print(f"뺄셈: a - b = {a - b}")           # [-3 -3 -3]
print(f"곱셈: a * b = {a * b}")           # [4 10 18]
print(f"나눗셈: a / b = {a / b}")         # [0.25 0.4  0.5]
print(f"거듭제곱: a ** 2 = {a ** 2}")     # [1 4 9]
print(f"나머지: a % 2 = {a % 2}")         # [1 0 1]
```

### 5.2 함수를 이용한 연산

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# NumPy 함수 사용
print(f"np.add(a, b) = {np.add(a, b)}")        # [5 7 9]
print(f"np.subtract(a, b) = {np.subtract(a, b)}")  # [-3 -3 -3]
print(f"np.multiply(a, b) = {np.multiply(a, b)}")  # [4 10 18]
print(f"np.divide(a, b) = {np.divide(a, b)}")      # [0.25 0.4  0.5]
print(f"np.power(a, 2) = {np.power(a, 2)}")        # [1 4 9]
```

### 5.3 2차원 배열 연산

```python
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

print(f"Array 1:\n{arr1}")
print(f"Array 2:\n{arr2}")

# 요소별 연산
print(f"요소별 덧셈:\n{arr1 + arr2}")
print(f"요소별 곱셈:\n{arr1 * arr2}")
```

### 5.4 행렬 곱셈 (Matrix Multiplication)

```python
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

# 행렬 곱셈 (내적)
matrix_product = np.dot(arr1, arr2)
print(f"행렬 곱셈 (np.dot):\n{matrix_product}")

# @ 연산자 사용 (Python 3.5+)
matrix_product2 = arr1 @ arr2
print(f"행렬 곱셈 (@):\n{matrix_product2}")

# matmul 함수 사용
matrix_product3 = np.matmul(arr1, arr2)
print(f"행렬 곱셈 (np.matmul):\n{matrix_product3}")
```

### 5.5 브로드캐스팅 (Broadcasting)

```python
# 서로 다른 형태의 배열 간 연산
arr = np.array([[1, 2, 3], [4, 5, 6]])
scalar = 10

print(f"Original Array:\n{arr}")
print(f"Scalar: {scalar}")
print(f"Array + Scalar:\n{arr + scalar}")  # 모든 요소에 10 더하기

# 1차원과 2차원 배열 간 연산
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_1d = np.array([10, 20, 30])

print(f"2D + 1D Broadcasting:\n{arr_2d + arr_1d}")
```

---

## 6. 통계 연산과 집계 함수

### 6.1 기본 집계 함수

```python
arr = np.array([[-1, 2, 3], [3, 4, 8]])
print(f"Original Array:\n{arr}")

# 전체 배열에 대한 통계
print(f"합계: {arr.sum()}")              # 19
print(f"평균: {arr.mean()}")             # 3.1666...
print(f"표준편차: {arr.std()}")          # 2.8674...
print(f"분산: {arr.var()}")              # 8.2222...
print(f"최댓값: {arr.max()}")            # 8
print(f"최솟값: {arr.min()}")            # -1
print(f"곱: {arr.prod()}")               # -576
```

### 6.2 위치 정보 함수

```python
arr = np.array([[-1, 2, 3], [3, 4, 8]])

# 최댓값/최솟값의 인덱스
print(f"최댓값 인덱스: {arr.argmax()}")   # 5 (평면화된 인덱스)
print(f"최솟값 인덱스: {arr.argmin()}")   # 0

# 다차원에서의 인덱스
max_index = np.unravel_index(arr.argmax(), arr.shape)
min_index = np.unravel_index(arr.argmin(), arr.shape)
print(f"최댓값 위치 (행, 열): {max_index}")  # (1, 2)
print(f"최솟값 위치 (행, 열): {min_index}")  # (0, 0)
```

### 6.3 기타 유용한 함수

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

print(f"정렬: {np.sort(arr)}")           # [1 1 2 3 4 5 6 9]
print(f"정렬 인덱스: {np.argsort(arr)}")  # [1 3 6 0 2 4 7 5]
print(f"유니크 값: {np.unique(arr)}")     # [1 2 3 4 5 6 9]
print(f"중위수: {np.median(arr)}")        # 3.5
print(f"백분위수(25%): {np.percentile(arr, 25)}")  # 1.75
```

---

## 7. Axis 개념 완전 정복

### 7.1 Axis 개념 이해

```python
arr = np.array([[-1, 2, 3], [3, 4, 8]])
print(f"Original Array:\n{arr}")
print(f"Shape: {arr.shape}")  # (2, 3) - 2행 3열
```

**Axis 방향 이해:**
- **axis=0**: 행 방향 (↕️) - 세로로 연산 (행들을 따라)
- **axis=1**: 열 방향 (↔️) - 가로로 연산 (열들을 따라)

### 7.2 Axis별 연산 예제

```python
arr = np.array([[-1, 2, 3], [3, 4, 8]])

# axis=0: 행 방향 연산 (각 열별로 계산)
print("=== axis=0 (행 방향, 세로 연산) ===")
print(f"열별 합계: {arr.sum(axis=0)}")      # [2, 6, 11]
print(f"열별 평균: {arr.mean(axis=0)}")     # [1.0, 3.0, 5.5]
print(f"열별 최댓값: {arr.max(axis=0)}")    # [3, 4, 8]

# axis=1: 열 방향 연산 (각 행별로 계산)
print("\n=== axis=1 (열 방향, 가로 연산) ===")
print(f"행별 합계: {arr.sum(axis=1)}")      # [4, 15]
print(f"행별 평균: {arr.mean(axis=1)}")     # [1.33, 5.0]
print(f"행별 최댓값: {arr.max(axis=1)}")    # [3, 8]
```

### 7.3 시각적 이해

```python
# 3차원 배열에서의 axis
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"3D Array Shape: {arr_3d.shape}")  # (2, 2, 2)

print(f"axis=0 sum: {arr_3d.sum(axis=0)}")  # 첫 번째 차원을 따라
print(f"axis=1 sum: {arr_3d.sum(axis=1)}")  # 두 번째 차원을 따라
print(f"axis=2 sum: {arr_3d.sum(axis=2)}")  # 세 번째 차원을 따라
```

### 7.4 실제 활용 예제

```python
# 학생별 과목 성적 데이터
scores = np.array([
    [85, 90, 78],  # 학생 1의 수학, 영어, 과학 점수
    [92, 88, 85],  # 학생 2의 점수
    [78, 85, 92],  # 학생 3의 점수
    [90, 87, 89]   # 학생 4의 점수
])

print("학생별 과목 성적:")
print(scores)

# 과목별 평균 (axis=0)
subject_avg = scores.mean(axis=0)
print(f"\n과목별 평균 점수: {subject_avg}")
print("수학: {:.1f}, 영어: {:.1f}, 과학: {:.1f}".format(*subject_avg))

# 학생별 평균 (axis=1)
student_avg = scores.mean(axis=1)
print(f"\n학생별 평균 점수: {student_avg}")
for i, avg in enumerate(student_avg, 1):
    print(f"학생 {i}: {avg:.1f}")
```

---

## 8. 실전 예제

### 8.1 데이터 분석 예제

```python
# 월별 매출 데이터 (4개 지점, 12개월)
sales_data = np.array([
    [120, 135, 142, 158, 165, 178, 185, 192, 175, 168, 152, 145],  # 지점 1
    [98, 112, 125, 138, 145, 159, 167, 174, 162, 155, 140, 128],   # 지점 2
    [87, 95, 108, 125, 135, 148, 156, 163, 151, 142, 125, 115],    # 지점 3
    [110, 125, 132, 149, 158, 172, 180, 187, 171, 164, 148, 138]   # 지점 4
])

print("월별 지점 매출 데이터:")
print(sales_data)

# 월별 전체 매출 (axis=0)
monthly_total = sales_data.sum(axis=0)
print(f"\n월별 전체 매출: {monthly_total}")
print(f"최고 매출 달: {monthly_total.argmax() + 1}월")
print(f"최고 매출액: {monthly_total.max()}")

# 지점별 연간 매출 (axis=1)
branch_annual = sales_data.sum(axis=1)
print(f"\n지점별 연간 매출: {branch_annual}")
best_branch = branch_annual.argmax() + 1
print(f"최고 실적 지점: 지점 {best_branch}")
print(f"최고 실적 지점 매출: {branch_annual.max()}")

# 전체 통계
print(f"\n=== 전체 통계 ===")
print(f"총 매출: {sales_data.sum()}")
print(f"평균 매출: {sales_data.mean():.2f}")
print(f"매출 표준편차: {sales_data.std():.2f}")
```

### 8.2 이미지 처리 시뮬레이션

```python
# 간단한 그레이스케일 이미지 시뮬레이션 (8x8 픽셀)
image = np.random.randint(0, 256, (8, 8))
print("Original Image (8x8 pixels):")
print(image)

# 이미지 통계
print(f"\n=== 이미지 통계 ===")
print(f"평균 밝기: {image.mean():.2f}")
print(f"최대 밝기: {image.max()}")
print(f"최소 밝기: {image.min()}")

# 이미지 처리
# 밝기 조절 (+50)
brighter = np.clip(image + 50, 0, 255)  # 0-255 범위로 제한
print(f"\n밝기 조절 후 평균: {brighter.mean():.2f}")

# 이미지 일부 추출 (4x4 크롭)
cropped = image[2:6, 2:6]
print(f"\n크롭된 이미지 (4x4):")
print(cropped)
```

### 8.3 수학 연산 예제

```python
# 벡터 연산
vector_a = np.array([3, 4, 0])
vector_b = np.array([1, 2, 2])

print(f"Vector A: {vector_a}")
print(f"Vector B: {vector_b}")

# 벡터 크기 (노름)
magnitude_a = np.linalg.norm(vector_a)
magnitude_b = np.linalg.norm(vector_b)
print(f"Vector A 크기: {magnitude_a:.2f}")
print(f"Vector B 크기: {magnitude_b:.2f}")

# 내적
dot_product = np.dot(vector_a, vector_b)
print(f"내적: {dot_product}")

# 외적 (3차원 벡터만)
cross_product = np.cross(vector_a, vector_b)
print(f"외적: {cross_product}")

# 코사인 유사도
cosine_similarity = dot_product / (magnitude_a * magnitude_b)
print(f"코사인 유사도: {cosine_similarity:.4f}")
```

---

## 9. 요약 및 핵심 포인트

### 9.1 NumPy 핵심 개념 요약

#### ✅ **NumPy의 장점**
1. **속도**: C로 구현되어 Python 리스트보다 훨씬 빠름
2. **메모리 효율성**: 연속된 메모리 공간 사용
3. **벡터화 연산**: 반복문 없이 전체 배열 연산 가능
4. **브로드캐스팅**: 다른 크기의 배열 간 연산 지원
5. **풍부한 함수**: 수학, 통계, 선형대수 함수 제공

#### 🔑 **핵심 개념**

| 개념 | 설명 | 예제 |
|------|------|------|
| **ndarray** | NumPy의 다차원 배열 객체 | `np.array([1,2,3])` |
| **shape** | 배열의 차원별 크기 | `(2, 3)` = 2행 3열 |
| **axis** | 연산을 수행할 차원 | `axis=0` (행), `axis=1` (열) |
| **broadcasting** | 크기가 다른 배열 간 연산 | `arr + 10` |
| **slicing** | 배열의 부분 선택 | `arr[1:3, 0:2]` |

#### 📊 **주요 함수별 정리**

**배열 생성:**
```python
np.array(), np.zeros(), np.ones(), np.full(), np.arange(), np.linspace()
```

**배열 정보:**
```python
.shape, .size, .ndim, .dtype, .itemsize
```

**통계 함수:**
```python
.sum(), .mean(), .std(), .var(), .max(), .min(), .argmax(), .argmin()
```

**연산 함수:**
```python
np.add(), np.subtract(), np.multiply(), np.divide(), np.dot(), np.matmul()
```

### 9.2 Axis 완전 이해

```python
# 2차원 배열에서 axis 이해
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# axis=0: ↓ 방향 (행을 따라, 각 열에 대해)
arr.sum(axis=0)  # [5, 7, 9] - 각 열의 합

# axis=1: → 방향 (열을 따라, 각 행에 대해)  
arr.sum(axis=1)  # [6, 15] - 각 행의 합
```

### 9.3 실무 활용 팁

#### 🎯 **성능 최적화**
1. **벡터화 사용**: 반복문 대신 배열 연산 사용
2. **적절한 dtype**: 필요한 만큼의 정밀도만 사용
3. **메모리 효율**: 불필요한 복사 피하기
4. **broadcasting 활용**: 명시적 반복 대신 브로드캐스팅 사용

#### 🔧 **디버깅 팁**
1. **shape 확인**: `print(array.shape)`으로 배열 구조 확인
2. **dtype 확인**: 예상치 못한 타입 변환 주의
3. **브로드캐스팅 오류**: 배열 크기 불일치 시 shape 확인
4. **axis 혼동**: 연산 결과가 예상과 다를 때 axis 값 재확인

#### 📝 **코딩 베스트 프랙티스**
1. **명확한 변수명**: `arr`, `matrix`, `data` 등 의미있는 이름 사용
2. **주석 추가**: 복잡한 배열 연산에는 설명 추가
3. **함수 분리**: 복잡한 배열 처리는 별도 함수로 분리
4. **에러 처리**: 배열 크기나 타입 검증 코드 추가

### 9.4 자주 하는 실수와 해결법

#### ❌ **흔한 실수들**

**1. axis 혼동**
```python
# 잘못된 이해
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr.sum(axis=0)  # [5, 7, 9] - "0번째 축을 제거"가 아니라 "0번째 축을 따라 연산"

# 올바른 이해: axis=0은 행 방향으로 연산 (각 열의 합)
```

**2. 얕은 복사 vs 깊은 복사**
```python
# 얕은 복사 (주의!)
arr1 = np.array([1, 2, 3])
arr2 = arr1  # 같은 메모리를 참조
arr2[0] = 999
print(arr1)  # [999, 2, 3] - arr1도 변경됨!

# 깊은 복사 (안전)
arr2 = arr1.copy()  # 새로운 배열 생성
arr2[0] = 999
print(arr1)  # [1, 2, 3] - arr1은 그대로
```

**3. 브로드캐스팅 오해**
```python
# 오류 발생하는 경우
arr1 = np.array([[1, 2, 3]])     # (1, 3)
arr2 = np.array([[1], [2]])      # (2, 1)
result = arr1 + arr2             # (2, 3) - 브로드캐스팅 됨!

# 의도하지 않은 결과가 나올 수 있으니 shape 항상 확인!
```

#### ✅ **해결 방법들**

**1. 디버깅용 유틸 함수**
```python
def debug_array(arr, name="Array"):
    """배열 정보를 출력하는 디버깅 함수"""
    print(f"=== {name} ===")
    print(f"Value:\n{arr}")
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")
    print(f"Dimension: {arr.ndim}")
    print("=" * 20)

# 사용 예시
arr = np.array([[1, 2], [3, 4]])
debug_array(arr, "Test Array")
```

**2. 안전한 배열 연산**
```python
def safe_array_operation(arr1, arr2, operation='add'):
    """안전한 배열 연산 함수"""
    # 타입 검사
    if not isinstance(arr1, np.ndarray) or not isinstance(arr2, np.ndarray):
        raise TypeError("Both inputs must be NumPy arrays")
    
    # shape 호환성 검사 (브로드캐스팅 고려)
    try:
        np.broadcast_arrays(arr1, arr2)
    except ValueError as e:
        raise ValueError(f"Array shapes are incompatible: {e}")
    
    # 연산 수행
    operations = {
        'add': np.add,
        'subtract': np.subtract,
        'multiply': np.multiply,
        'divide': np.divide
    }
    
    if operation not in operations:
        raise ValueError(f"Unsupported operation: {operation}")
    
    return operations[operation](arr1, arr2)
```

### 9.5 학습 체크리스트

#### 🎯 **기본 개념 (✅ 체크해보세요!)**
- [ ] NumPy와 Python List의 차이점을 설명할 수 있다
- [ ] ndarray의 주요 속성들(shape, size, ndim, dtype)을 안다
- [ ] 다양한 방법으로 배열을 생성할 수 있다
- [ ] 배열의 인덱싱과 슬라이싱을 할 수 있다

#### 🔧 **중급 개념**
- [ ] axis 개념을 완전히 이해했다
- [ ] 브로드캐스팅이 무엇인지 안다
- [ ] 벡터화 연산의 장점을 안다
- [ ] 요소별 연산과 행렬 곱셈의 차이를 안다

#### 🚀 **고급 활용**
- [ ] 실제 데이터를 NumPy로 처리할 수 있다
- [ ] 성능을 고려한 코드를 작성할 수 있다
- [ ] 복잡한 배열 조작을 할 수 있다
- [ ] 다른 라이브러리(pandas, matplotlib)와 연동할 수 있다

### 9.6 다음 학습 방향

#### 📈 **연관 라이브러리**
1. **Pandas**: 데이터프레임 처리 (NumPy 기반)
2. **Matplotlib/Seaborn**: 데이터 시각화
3. **SciPy**: 과학 계산 라이브러리
4. **Scikit-learn**: 머신러닝 (NumPy 배열 사용)

#### 🎯 **심화 학습 주제**
1. **선형대수**: `np.linalg` 모듈
2. **푸리에 변환**: `np.fft` 모듈
3. **랜덤 생성**: `np.random` 모듈
4. **파일 입출력**: `np.save`, `np.load`

#### 💡 **실습 프로젝트 아이디어**
1. **이미지 처리**: 필터 적용, 크기 조절
2. **데이터 분석**: CSV 파일 분석, 통계 계산
3. **수치 해석**: 방정식 해, 최적화 문제
4. **신호 처리**: 오디오 데이터 분석

---

## 10. 실습 코드 모음

### 10.1 종합 실습 1: 학생 성적 관리 시스템

```python
import numpy as np

# 데이터 준비 (5명 학생, 4과목 성적)
students = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve']
subjects = ['Math', 'English', 'Science', 'History']

# 성적 데이터 (5x4 배열)
scores = np.array([
    [85, 92, 78, 88],  # Alice
    [90, 85, 95, 82],  # Bob
    [78, 88, 85, 90],  # Charlie
    [92, 90, 88, 85],  # Diana
    [88, 78, 92, 87]   # Eve
])

print("=== 학생 성적 관리 시스템 ===")
print("\n1. 전체 성적표:")
print("Students:", students)
print("Subjects:", subjects)
print("Scores:\n", scores)

print("\n2. 과목별 통계:")
subject_stats = {
    'mean': scores.mean(axis=0),
    'max': scores.max(axis=0),
    'min': scores.min(axis=0),
    'std': scores.std(axis=0)
}

for i, subject in enumerate(subjects):
    print(f"{subject:8}: 평균 {subject_stats['mean'][i]:.1f}, "
          f"최고 {subject_stats['max'][i]}, "
          f"최저 {subject_stats['min'][i]}, "
          f"표준편차 {subject_stats['std'][i]:.1f}")

print("\n3. 학생별 통계:")
student_stats = {
    'total': scores.sum(axis=1),
    'mean': scores.mean(axis=1),
    'max': scores.max(axis=1),
    'min': scores.min(axis=1)
}

for i, student in enumerate(students):
    print(f"{student:8}: 총점 {student_stats['total'][i]}, "
          f"평균 {student_stats['mean'][i]:.1f}, "
          f"최고 {student_stats['max'][i]}, "
          f"최저 {student_stats['min'][i]}")

print("\n4. 전체 분석:")
print(f"전체 평균: {scores.mean():.2f}")
print(f"최고 성적 학생: {students[student_stats['mean'].argmax()]}")
print(f"최고 성적: {student_stats['mean'].max():.2f}")
print(f"가장 어려운 과목: {subjects[subject_stats['mean'].argmin()]}")
print(f"가장 쉬운 과목: {subjects[subject_stats['mean'].argmax()]}")

# 성적 등급 부여
def get_grade(score):
    if score >= 90: return 'A'
    elif score >= 80: return 'B'
    elif score >= 70: return 'C'
    elif score >= 60: return 'D'
    else: return 'F'

print("\n5. 등급 분포:")
grades = np.vectorize(get_grade)(scores)
unique, counts = np.unique(grades, return_counts=True)
for grade, count in zip(unique, counts):
    print(f"등급 {grade}: {count}명")
```

### 10.2 종합 실습 2: 간단한 이미지 필터

```python
import numpy as np

# 간단한 8x8 그레이스케일 이미지 생성
np.random.seed(42)  # 재현 가능한 결과를 위해
original_image = np.random.randint(0, 256, (8, 8))

print("=== 이미지 처리 시스템 ===")
print("\n1. 원본 이미지 (8x8):")
print(original_image)

# 이미지 통계
print(f"\n2. 이미지 정보:")
print(f"크기: {original_image.shape}")
print(f"픽셀 수: {original_image.size}")
print(f"평균 밝기: {original_image.mean():.2f}")
print(f"최대 밝기: {original_image.max()}")
print(f"최소 밝기: {original_image.min()}")
print(f"표준편차: {original_image.std():.2f}")

# 3. 다양한 필터 적용
print("\n3. 필터 적용 결과:")

# 밝기 조절
brighter = np.clip(original_image + 50, 0, 255)
darker = np.clip(original_image - 50, 0, 255)
print(f"밝기 +50 후 평균: {brighter.mean():.2f}")
print(f"밝기 -50 후 평균: {darker.mean():.2f}")

# 대비 조절
high_contrast = np.clip(original_image * 1.5, 0, 255).astype(int)
low_contrast = np.clip(original_image * 0.5, 0, 255).astype(int)
print(f"높은 대비 후 표준편차: {high_contrast.std():.2f}")
print(f"낮은 대비 후 표준편차: {low_contrast.std():.2f}")

# 이진화 (임계값 128)
threshold = 128
binary = np.where(original_image > threshold, 255, 0)
print(f"이진화 후 고유값: {np.unique(binary)}")

# 히스토그램 분석 (간단한 구간별 픽셀 수)
bins = [0, 64, 128, 192, 256]
hist = np.histogram(original_image, bins=bins)[0]
print("\n4. 밝기 히스토그램:")
for i, count in enumerate(hist):
    print(f"{bins[i]:3d}-{bins[i+1]-1:3d}: {count:2d}픽셀")

# ROI (Region of Interest) 추출
roi = original_image[2:6, 2:6]  # 중앙 4x4 영역
print(f"\n5. 중앙 4x4 영역 평균 밝기: {roi.mean():.2f}")
print("ROI:")
print(roi)
```

### 10.3 종합 실습 3: 수학적 계산

```python
import numpy as np

print("=== 수학적 계산 종합 실습 ===")

# 1. 벡터 연산
print("\n1. 벡터 연산:")
v1 = np.array([3, 4, 5])
v2 = np.array([1, 2, 2])

print(f"벡터 v1: {v1}")
print(f"벡터 v2: {v2}")
print(f"내적: {np.dot(v1, v2)}")
print(f"외적: {np.cross(v1, v2)}")
print(f"v1 크기: {np.linalg.norm(v1):.3f}")
print(f"v2 크기: {np.linalg.norm(v2):.3f}")

# 코사인 유사도
cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
print(f"코사인 유사도: {cos_sim:.4f}")

# 2. 행렬 연산
print("\n2. 행렬 연산:")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"행렬 A:\n{A}")
print(f"행렬 B:\n{B}")
print(f"A + B:\n{A + B}")
print(f"A * B (요소별):\n{A * B}")
print(f"A @ B (행렬곱):\n{A @ B}")
print(f"A의 전치:\n{A.T}")
print(f"A의 행렬식: {np.linalg.det(A):.3f}")

try:
    print(f"A의 역행렬:\n{np.linalg.inv(A)}")
except np.linalg.LinAlgError:
    print("A는 특이행렬입니다 (역행렬이 존재하지 않음)")

# 3. 통계적 계산
print("\n3. 통계적 계산:")
data = np.random.normal(100, 15, 1000)  # 평균 100, 표준편차 15인 정규분포 데이터

print(f"데이터 크기: {data.size}")
print(f"평균: {data.mean():.2f}")
print(f"중위수: {np.median(data):.2f}")
print(f"표준편차: {data.std():.2f}")
print(f"분산: {data.var():.2f}")
print(f"최솟값: {data.min():.2f}")
print(f"최댓값: {data.max():.2f}")
print(f"25% 백분위수: {np.percentile(data, 25):.2f}")
print(f"75% 백분위수: {np.percentile(data, 75):.2f}")

# 4. 다항식 연산
print("\n4. 다항식 연산:")
# 다항식: 2x^2 + 3x + 1
coeffs = [2, 3, 1]  # 높은 차수부터
x_values = np.linspace(-5, 5, 11)
y_values = np.polyval(coeffs, x_values)

print(f"다항식: 2x² + 3x + 1")
print("x값들:", x_values)
print("y값들:", y_values.round(2))

# 근 찾기
roots = np.roots(coeffs)
print(f"다항식의 근: {roots}")

# 5. 삼각함수
print("\n5. 삼각함수:")
angles = np.linspace(0, 2*np.pi, 8)
print("각도(라디안):", angles.round(3))
print("sin 값:", np.sin(angles).round(3))
print("cos 값:", np.cos(angles).round(3))
print("tan 값:", np.tan(angles).round(3))
```

---

## 📚 마무리

이 문서는 NumPy의 핵심 개념부터 실전 활용까지 체계적으로 정리한 완전한 학습 가이드입니다. 

**🎯 활용 방법:**
1. **기초 학습**: 1-4장으로 NumPy 기본기 다지기
2. **실전 연습**: 5-8장으로 실무 능력 키우기  
3. **심화 학습**: 9-10장으로 전문성 높이기
4. **참고 자료**: 필요할 때마다 해당 섹션 참조

**📖 계속 학습하려면:**
- 공식 문서: https://numpy.org/doc/
- 튜토리얼: https://numpy.org/learn/
- 실습 환경: Jupyter Notebook 또는 Google Colab

NumPy를 마스터하면 데이터 사이언스, 머신러닝, 과학 계산의 든든한 기초가 될 것입니다! 🚀