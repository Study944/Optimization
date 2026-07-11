"""差分进化算法"""
import numpy as np
import benchMark.fitness_func as cec_function
import Differential_Evolution as MyDE
import lshape.LSHADE as MyLSHADE

def de_test():
    # 1.获取6个CEC测试函数
    func_list = cec_function.get_all_func()
    # 2.调用DE分别计算
    for func in func_list:
        dim = 10
        de = MyDE.DE(dim=dim)
        limits = np.array([[-5.12, 5.12]] * dim)
        x,fit = de.optimize(func, limits)
        print(f'最优位置{x}，最优值{fit}')

def lshade_test():
    # 1.获取6个CEC测试函数
    func_list = cec_function.get_all_func()
    # 2.调用DE分别计算
    for func in func_list:
        dim = 10
        ls = MyLSHADE.LSHADE(dim=dim)
        limits = np.array([[-5.12, 5.12]] * dim)
        x,fit = ls.optimize(func, limits)
        print(f'最优位置{x}，最优值{fit}')

if __name__ == '__main__':
    print("DE执行流程")
    de_test()
    print("LSHADE执行流程")
    lshade_test()

