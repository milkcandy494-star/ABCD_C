# -*- coding: utf-8 -*-

def print_gugudan(num):
    """주어진 숫자의 구구단을 출력하는 함수"""
    print(f"\n=== {num}단 ===")
    for i in range(1, 10):
        result = num * i
        print(f"{num} × {i} = {result}")
    print()

def main():
    """메인 함수"""
    print("구구단 프로그램입니다!")
    
    while True:
    
        try:
            # 사용자로부터 구구단 수 입력받기
            num = int(input("몇 단을 출력하시겠습니까? (종료하려면 0 입력): "))
            
            # 0 입력 시 프로그램 종료
            if num == 0:
                print("프로그램을 종료합니다.")
                break
            
            # 1~9단만 허용
            if 1 <= num <= 9:
                print_gugudan(num)
            else:
                print("1부터 9까지의 숫자만 입력해주세요.")
                
        except ValueError:
            print("올바른 숫자를 입력해주세요.")
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            break

if __name__ == "__main__":
    main()