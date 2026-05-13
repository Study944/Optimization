# 粒子群算法 `PSO`

## `PSO` 实现

使用 PSO 计算 f(x) = x * sin(x) * cos(2x) − 2x * sin(3x) + 3x * sin(4x)在[0,50]的最小值。

![img.png](img/img.png)
图一：数据展示
![img_1.png](img/img1.png)
图二：中间轮次
![img_2.png](img/img2.png)
图三：结果

使用 PSO 计算 f(x,y) = 20 + x**2 + y**2 − 10*cos(2𝜋x) − 10*cos(2𝜋𝑦)在[−5.12,5.12]的最小值

![img.png](img/img3.png)
图四：数据展示
![img.png](img/img4.png)
图五：中间轮次
![img.png](img/img5.png)
图六：结果

测试不同惯性系数$w$、个体学习因子$c1$和社会学系因子$c2$

![img.png](img/img6.png)