import tspdata.Load_Data as tspData
import Ant_Colony_Optimization as myACO

if __name__ == '__main__':
    # 1.加载tsp数据集
    dataset_registry = tspData.load_all_tsp_data()
    # 2.初始化ACO对象
    aco = myACO.ACO()
    # 3.测试
    for name, data in dataset_registry.items():
        best_length, best_path = aco.optimize(data["matrix"])
        print(f'数据集：{name}，最短路径长度：{data["optimal"]}，ACO路径长度：{best_length}')
