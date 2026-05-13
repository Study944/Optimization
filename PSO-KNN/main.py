from experiments.adult.run_adult import run_adult
from experiments.adult.run_adult_dwknn import run_adult_dwknn
from experiments.adult.run_adult_weight import run_adult_weight
from experiments.cancer.run_cancer_dwknn import run_cancer_dwknn
from experiments.cancer.run_cancer_weight import run_cancer_weight
from experiments.iris.run_iris import run_iris
from experiments.iris.run_iris_dwknn import run_iris_dwknn
from experiments.iris.run_iris_weight import run_iris_weight
from experiments.wine.run_wine import run_wine
from experiments.cancer.run_cancer import run_cancer
from experiments.wine.run_wine_dwknn import run_wine_dwknn
from experiments.wine.run_wine_weight import run_wine_weight

if __name__ == '__main__':
    print("版本1.1(PSO:k值)测试结果：----------------------------------------------------------")
    print("开始运行 Iris 数据集 PSO-KNN 优化实验...")
    run_iris()

    print("开始运行 Wine 数据集 PSO-KNN 优化实验...")
    run_wine()

    print("开始运行 Breast-Cancer 数据集 PSO-KNN 优化实验...")
    run_cancer()

    print("开始运行 Adult 数据集 PSO-KNN 优化实验...")
    run_adult()

    print("版本2.1(PSO:k值)+(DWKNN)测试结果----------------------------------------------------------")
    print("开始运行 Iris 数据集 PSO-DWKNN 优化实验...")
    run_iris_dwknn()

    print("开始运行 Wine 数据集 PSO-DWKNN 优化实验...")
    run_wine_dwknn()

    print("开始运行 Breast-Cancer 数据集 PSO-DWKNN 优化实验...")
    run_cancer_dwknn()

    print("开始运行 Adult 数据集 PSO-DWKNN 优化实验...")
    run_adult_dwknn()

    print("版本3.1(PSO:k值,特征权重)测试结果----------------------------------------------------------")
    print("开始运行 Iris 数据集 PSO-KNN 优化实验...")
    run_iris_weight()

    print("开始运行 Wine 数据集 PSO-KNN 优化实验...")
    run_wine_weight()

    print("开始运行 Breast-Cancer 数据集 PSO-KNN 优化实验...")
    run_cancer_weight()

    print("开始运行 Adult 数据集 PSO-KNN 优化实验...")
    run_adult_weight()
