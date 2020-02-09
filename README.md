# Robot-Planning-and-Control
每周更新自动驾驶领域论文或相关算法实现，路径规划为主。
QQ交流群：861253468



# 强化学习基础







## 2019-Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model
相关视频：

- https://www.youtube.com/watch?v=hYV4-m7_SK8

- https://www.youtube.com/watch?v=We20YSAJZSE



构建具有规划能力的智能体一直是追求人工智能的主要挑战之一。基于树的规划方法在具有挑战性的领域取得了巨大的成功，例如象棋和围棋，在这些领域有一个完美的模拟器。然而，在现实问题中，控制环境的动力往往是复杂和未知的。在这项工作中，我们提出了MuZero算法，该算法通过将基于树的搜索与学习模型相结合，在一系列具有挑战性和视觉复杂性的领域中实现超人的性能，而不需要了解它们的底层动力学。MuZero学习了一个模型，当迭代应用时，它预测与计划最直接相关的数量：奖励、行动选择策略和价值函数。当在57个不同的Atari游戏上进行评估时，我们的新算法达到了一个新的水平，这是一个测试人工智能技术的标准视频游戏环境，在这个环境中，基于模型的规划方法一直在挣扎。在进行评估时，国际象棋和shogi在不了解游戏规则的情况下，MuZero与游戏规则提供的AlphaZero算法的超人性能相匹配。











### 2017-NIPS-Imagination-Augmented Agents for Deep Reinforcement Learning

相关视频：

- https://www.youtube.com/watch?v=xp-YOPcjkFw

- https://www.youtube.com/watch?v=agXIYMCICcc



#### 1 Introduction

在使用深度神经网络和**无模型强化学**习（RL）为许多领域开发有能力的代理方面取得了进展，其中原始观测直接映射到值或动作。然而，这种方法通常需要大量的训练数据，并且所得到的策略不容易推广到相同环境中的新任务，因为它缺乏一般智力的行为灵活性构成。

**基于模型的RL**旨在通过赋予代理一个由过去经验合成的世界模型来解决这些缺点。通过使用内部模型对未来进行推理（这里也称为想象），代理人可以寻求积极的结果，同时避免现实环境中反复试验的不利后果，包括做出不可逆转、糟糕的决定。即使该模型需要首先学习，它也可以更好地跨状态进行泛化，在同一环境中跨任务保持有效，并利用额外的无监督学习信号，从而最终提高数据效率。基于模型的方法的另一个吸引人之处是它们能够通过增加内部模拟的数量的更多计算来扩展性能。

**最近的成功主要来自无模型方法**。在这样的领域中，基于模型的代理采用标准规划方法的性能通常遭受由函数逼近引起的模型误差。这些错误在规划过程中会加剧，导致过于乐观或性能差。

**我们试图通过提出想象增强来解决这个缺点**，它通过“学习解释”其不完美的预测来使用近似的环境模型。我们的算法可以直接在低层的观测数据上训练，而不需要太多的领域知识，类似于最近的无模型成功。在不对环境模型的结构及其可能存在的缺陷作出任何假设的情况下，我们的方法以端到端的方式学习，从模型模拟中提取有用的知识，特别是不完全依赖模拟收益。这使得代理可以从基于模型的想象中获益，而不必像传统的基于模型的规划那样陷入陷阱。我们证明，在包括Sokoban在内的各个领域中，我们的方法比无模型基线执行得更好。即使在模型不完善的情况下，它也能以较少的数据获得更好的性能，这是实现基于模型的RL的重要一步。



#### 2 The I2A architecture

![image-20200208111215557](/home/lichunhong/.config/Typora/typora-user-images/image-20200208111215557.png)





# 基于学习的决策



## 强化学习

### 2019-Predictive Trajectory Planning in Situations with Hidden Road Users Using Partially Observable Markov Decision Processes

摘要-近年来，仅基于传感器测量的最先进的紧急制动辅助系统大大减少了交通事故和人员伤亡。为了能够对因传感器限制或遮挡而避开车辆视野的道路使用者做出反应，提出了一种在自主车辆决策过程中预测遮挡区域潜在隐藏交通参与者的方法。采用**部分可观测的马尔可夫决策过程**来**确定车辆的纵向运动**。使用车辆的视野进行观察。因此，根据当前或预测的环境，使用传感器设置的通用模型计算视野。这样，车辆既可以观察到它检测到先前隐藏的道路使用者，也可以接收到道路畅通的信息。总的来说，这使得车辆能够更好地预测未来的发展。因此，需要对可能位于隐蔽区域的车辆进行假设。我们将在两个场景中演示该方法。首先在一个场景中，车辆必须以最少的动作谨慎地驶入交叉口，其次在城市交通的典型场景中。评价结果表明，该方法能够正确预测隐藏的道路使用者，并采取相应的措施。

- 这篇文章获得了**IV19 Best Paper Award**，可见POMDP在自动驾驶领域的潜力。
- 本文能够正确预测隐藏的道路使用者，相当于进行了风险评估，对车辆加减速进行控制。



提出了一种概率规划方法，能够在预期的规划过程中安全地处理上述情况。我们使用部分可观察马尔可夫决策过程（POMDP）来解决潜在道路使用者的挑战。在规划过程中，自主车辆能够根据当前和预测的环境考虑未来的观测。为了确定车辆将看到什么，我们使用车辆传感器设置的通用表示来计算其视野。总的来说，视野和对隐藏车辆的假设的结合导致了一个更具前瞻性和前瞻性的规划过程。

![image-20200207154739565](/home/lichunhong/.config/Typora/typora-user-images/image-20200207154739565.png)





## Learning Driver Behavior Models from Traffic Observations for Decision Making and Planning

摘要：对于复杂的驾驶员辅助系统和自动驾驶来说，随着时间的推移估计和预测交通状况是必不可少的能力。当需要更长的预测范围时，例如在决策或运动规划中，在不牺牲稳健性和安全性的前提下，不完全环境感知和随时间的随机情况发展所引起的不确定性是不可忽略的。**建立驾驶员与环境、道路网络和其他交通参与者相互作用的一致概率模型**是一个复杂的问题。本文通过建立描述驾驶员行为和计划的层次动态贝叶斯模型，对驾驶员的决策过程进行建模。这样，所有抽象级别的过程中的不确定性都可以用数学上一致的方式处理。由于驾驶员行为难以建模，我们提出了一种学习交通参与者行为的连续、非线性、上下文相关模型的方法。我们提出了一个期望最大化（EM）的方法来学习集成在DBN的模型从未标记的观测。实验表明，与只考虑车辆动力学的标准模型相比，该模型在估计和预测精度上有了显著提高。最后，提出了一种新的自主驾驶策略决策方法。它是基于一个连续的部分可观测马尔可夫决策过程（POMDP），使用该模型进行预测。





# 基于规则的决策



## 2018-Reachability-based Decision Making for City Driving

针对具有高级驾驶辅助和自动化特征的车辆，设计了一种离散决策算法。我们将系统建模为一个混合自动机，其中自动机中离散模式之间的转换对应于驱动模式决策，并开发了一种基于前后可达集的模式转换时间确定方法。该算法既可以作为一个独立的组件，也可以作为一种方法来指导底层的运动规划器获得安全的参考轨迹。在一定的假设条件下，该算法保证了城市交通的安全性和生动性，并通过计算机仿真验证了算法的有效性。











![image-20200204121050465](/home/lichunhong/.config/Typora/typora-user-images/image-20200204121050465.png)

视频参考：https://www.youtube.com/watch?v=uLOsCZ4s03U



## 2019-Trajectory Optimization and Situational Analysis Framework for Autonomous Overtaking with Visibility Maximization

### IV. BEHAVIORAL PLANNER

![image-20200121135656035](./imags/image-20200121135656035.png)



- F: Follow ego-lane  沿车道行驶，需要约束路径偏离
- V: Visibility Maximization　最大化视野范围，需要放宽路径偏离约束
- O: Overtake　超车，需要计算超车路径
- M: Merge back　切换回原先的车道，需要计算换道路径
- W: Wait:　减速直至停止，观测环境变化



σ1 : Obstacle to be overtaken in ego lane detected.
σ2 : Visibility and overtaking time is sufficient / no feasible ego lane trajectory.
σ3 : Complete occlusion.
σ4 : Overtaking maneuver is completed.
σ5 : Incoming traffic in opposite lane detected and overtaking time is insufficient.
σ6 : Incoming traffic is cleared, and sufficiency criteria not yet fulfilled.
σ7 : Incoming traffic in opposite lane detected and overtaking time is insufficient.
σ8 : Incoming traffic is cleared, and sufficiency criteria are fulfilled.
σ9 : Incoming traffic in opposite lane detected and overtaking time is insufficient.
σ10 : Incoming traffic is cleared, and overtaking maneuver is completed or canceled.
σ11 : Merging maneuver is completed.

### V. TRAJECTORY GENERATION

#### A. Vehicle Model

#### B. Path Representation and Tracking

#### C. Road Boundaries



![image-20200121140821766](./imags/image-20200121140821766.png)



#### D. Obstacle Representation



#### E. Visibility Maximization

以“视野角”的大小表示感知情况，评估风险，并作为目标函数进行优化

![image-20200121143145968](./imags/image-20200121143145968.png)





<img src="./imags/image-20200121142835394.png" alt="image-20200121142835394" style="zoom:80%;" />





#### F. MPC Formulation



### VI. SITUATIONAL ANALYSIS FRAMEWORK

#### A. Occupancy of Other Traffic Participants





![image-20200121154427920](./imags/image-20200121154427920.png)

超车的最佳时间为$t_{overtake}$





#### B. Information Sufficiency

![image-20200121151957801](./imags/image-20200121151957801.png)



信息充分性：如果车辆在sufficiency line之前，信息是充分的，视野良好；反之，视野受限，风险增加。

$S_{sufficient}$的值决定了行为的激进/保守程度，这里选取$S_{sufficient}$为一个车长的距离(4m)。



#### C. Overtaking Maneuver Risk Assessment



TODO：

查看文献27：dynamic virtual bumper



![image-20200121154656833](./imags/image-20200121154656833.png)



超车需要规划出一条超车参考路线，需要综合考虑车辆运动学和动力学约束。







### VII. SIMULATION RESULTS







## 2014-A Behavioral Planning Framework for Autonomous Driving.pdf

![image-20200205155523496](/home/lichunhong/.config/Typora/typora-user-images/image-20200205155523496.png)





#　轨迹预测

标题：Trajectron++: Multi-Agent Generative Trajectory Forecasting With Heterogeneous Data for Control.pdf

作者：Tim Salzmann　 Boris Ivanovic　Punarjay Chakravarty　Marco Pavone

团队：Autonomous Systems Lab, Stanford University 　　Ford Greenfield Labs

摘要：对人类在环境中的运动进行推理是实现安全的、具有社会意识的机器人导航的重要前提。因此，多智能体行为预测已经成为自动驾驶汽车等现代人机交互系统的核心组成部分。虽然存在多种用于轨迹预测的方法，但它们中的许多仅用一种语义类的智能体进行评估，并且仅使用先前的轨迹信息，忽略了从通用传感器到自治系统的在线可用的大量信息。为此，我们提出了Trajectron++，这是一个模块化的、图结构的递归模型，它可以在包含异构数据（如语义图和相机图像）的同时，预测具有不同语义类的一般智能体的轨迹。我们的模型与机器人规划和控制框架紧密结合，能够生成对主体运动规划产生重要影响的预测。我们在几个具有挑战性的现实世界轨迹预测数据集上演示了我们的模型的性能，其性能超过了一系列最先进的确定性和生成性方法。



# 基于学习的运动规划

## 深度学习





## 强化学习

### 2020-Survey of Deep Reinforcement Learning for Motion Planning of Autonomous Vehicles

**摘要：**近年来，自主汽车领域的学术研究在传感器技术、V2X通信、安全、安保、决策、控制，甚至法律和标准化规则等方面都得到了广泛的应用。除了经典的控制设计方法外，人工智能和机器学习方法几乎在所有这些领域都有应用。研究的另一部分集中在运动规划的不同层次，如战略决策、轨迹规划和控制。机器学习本身已经发展出一系列的技术，本文描述了其中一个领域，深度强化学习（DRL）。本文对分层运动规划问题进行了深入的研究，并介绍了DRL的基本原理。设计这样一个系统的主要要素是环境的建模、建模抽象、状态和感知模型的描述、适当的奖励和底层神经网络的实现。本文介绍了车辆模型、仿真可能性和计算要求。给出了不同层次和观测模型的战略决策，如连续和离散状态表示、基于网格和基于摄像机的解决方案。本文综述了由自动驾驶的不同任务和层次（如跟车、车道保持、轨迹跟踪、合并或在密集交通中驾驶）系统化的最新解决方案。最后，讨论了开放性问题和未来的挑战。

**标签：**综述性文章  深度强化学习



#### I. INTRODUCTION

**A. The Hierarchical Classification of Motion Planning for Autonomous Driving**

自动驾驶车辆运动规划的层次结构：路线规划、行为规划、运动规划、反馈控制

![image-20200207130739007](/home/lichunhong/.config/Typora/typora-user-images/image-20200207130739007.png)

**B. Reinforcement Learning**









# 基于人工势场的运动规划



# 基于采样的运动规划

## A*及其变种

参考简介：https://www.jianshu.com/p/a3951ce7574d

https://www.cnblogs.com/Leonhard-/p/6866070.html

https://www.cnblogs.com/yangrouchuan/p/6373285.html

https://blog.csdn.net/lqzdreamer/article/details/85108310



A* :

Weighted A*:

ARA\*: anytime repairing A\*

LPA*：Lifelong Planning A\*

D\*:D\*是动态A\*（[D-Star](https://baike.baidu.com/item/D-Star), Dynamic A*）

D* Lite:

Focussed D*:

Field D*:

Theta\*:算法可以优化A\*​的路径，但是如果A*算法部分做的不够好的话，theta\*效果会大打折扣。

LazyTheta*:











## Lattice



### 2015-Real-time motion planning methods for autonomous on-road driving: State-of-the-art and future research directions
综述文章







### 2014-Trajectory Planning for BERTHA -a Local, Continuous Method

摘要：本文在总结前人研究成果的基础上，提出了在完全自主完成柏莎-奔驰纪念路线103公里的车辆上进行轨迹规划的策略。我们提出一个由变分公式导出的局部连续方法。解的轨迹是一个目标函数的约束极值，该目标函数用于表达动态可行性和舒适性。静态和动态障碍物约束以多边形的形式合并。这些约束经过精心设计，以确保解收敛到单个全局最优解。

#### II. RELATED WORK

**A. Preliminaries**

**B. Objective function**

代价函数组成：

<img src="/home/lichunhong/.config/Typora/typora-user-images/image-20200117194211159.png" alt="image-20200117194211159" style="zoom:80%;" />



**C. Constraint functions**

约束项：内部约束、外部约束

内部约束：最大曲率、最大加速度等

外部约束：物体碰撞

![image-20200117194511139](/home/lichunhong/.config/Typora/typora-user-images/image-20200117194511139.png)

进行物体碰撞检测，自车用多个圆表示，障碍物用多边形表示。



**D. Building constraint polygons from sensor data**





静态障碍物：left-right decision 根据简单的几何结构，将障碍物分配到左右两边

![image-20200117202900285](/home/lichunhong/.config/Typora/typora-user-images/image-20200117202900285.png)



动态障碍物：分为迎面车辆，划分在右侧；超车车辆，划分在左侧

移动障碍物预测：常速并且与右边界保持固定的距离。



**E. Distance function**

//TODO

基于优化的路径选择

![image-20200117205411907](/home/lichunhong/.config/Typora/typora-user-images/image-20200117205411907.png)

**F. Re-planning scheme**

![image-20200117204615577](/home/lichunhong/.config/Typora/typora-user-images/image-20200117204615577.png)

**G. Constrained optimization**

//TODO















### 2011-Motion Planning for Autonomous Driving with a Conformal Spatiotemporal Lattice
摘要-我们提出了一种适用于公路自主驾驶的运动规划器，它采用了为行星漫游者导航而开创的状态格框架，以适应公共道路的结构化环境。本文的主要贡献在于提出了一种搜索空间表示方法，使得搜索算法能够系统有效地实时地探索时空维度。这使得低级planner，在有其他车辆在场的情况下，能够致力于规划跟随领先车辆、改变车道和绕过障碍物。我们证明了我们的算法可以很容易地在**GPU上加速**，并在自主乘用车上进行了演示。



#### I. INTRODUCTION

我们的计划者还使用了一种**新的时空搜索图**，它结合了沿选定空间维度的精确约束满足和沿时间维度的状态的分辨率等价剪枝，其结果是，在不过度增加状态空间大小的情况下，可以检查时间和速度的大量变化。

构建一个能够在复杂环境中智能操作的规划师的典型工作依赖于将规划解决方案分解为一个规划师层次结构，这些规划师依次对搜索空间进行更具体的表示，并在时间上进行更精细的离散化。层次结构中的每个计划者必须具有其他计划者的模型。这些模型必然有缺陷，否则分解将是多余的。规划者之间对行为的期望不匹配会导致计划动作的不稳定，特别是当上级规划者向下级规划者发出的命令不可行时。我们提出的规划框架通过在较低的层次上承担更多的责任来缓解这个问题，这些责任可以在尝试执行行动之前决定行动是否可行。

例如，通过在**统一优化框架内进行规划**，我们的规划者能够在没有特定行为指示的情况下，决定是保持在慢行交通后面的距离，还是改变车道以通过慢行交通。



#### II. RELATED WORK



#### III. METHOD



**A. Paths and Trajectories**

**B. Spatiotemporal Lattice**

每个顶点的状态向量表示为 $$(x,y,a,θ,κ,[ti, ti+1),[vj, vj+1))$$。在Frenet坐标系中，可简化表示为五维向量$$(s,l,a,[ti, ti+1),[vj, vj+1))$$。



搜索算法：

![image-20200206110050966](/home/lichunhong/.config/Typora/typora-user-images/image-20200206110050966.png)





**C. Why include acceleration in the state space?**

![image-20200205105802799](/home/lichunhong/.config/Typora/typora-user-images/image-20200205105802799.png)

从相同顶点出发的轨迹末端状态可能落在相同的时间速度晶格上，使用加速度维度区分轨迹。最终的轨迹拥有更一致的加速度剖面。

**D. Cost function**

代价函数分成两部分：

由于多个轨迹使用相同的基本路径，因此在使用该路径的各种轨迹之前，仅依赖于$$（x，y，θ，k）$$的代价函数项被计算。然后根据每条轨迹计算$$a，t，v$$的项。



**E. Picking the best final state**

代价函数：最小化轨迹成本的加权总和上+进一步行驶的奖励+花费额外时间的惩罚。

![image-20200206103655760](/home/lichunhong/.config/Typora/typora-user-images/image-20200206103655760.png)

然后，从$$n_f$$到起始状态反向追溯，重构最优轨迹。



**F. World Representation**

静态障碍物的表示：通过离散化的$$(x,y)$$空间，将静态障碍物放到查找表中。

动态障碍物的表示：离散化三维$$(x,y,t)$$空间，将移动障碍物放入表格中。

障碍物用矩形框表示，在每次采样时执行一次查表操作。



#### IV. GPU ACCELERATION

GPU可以用来加速算法。当最坏的搜索情况发生时，不可避免要遍历整个图。所以我们必须搜索整个图。





#### V. EXPERIMENTAL RESULTS

分别在CPU和GPU上测试了完整的规划所需时间，如下表所示。

![image-20200205114538777](/home/lichunhong/.config/Typora/typora-user-images/image-20200205114538777.png)

该planner展示了良好的轨迹规划特性，例如过弯道时，能够规划出合理的速度和加速度。在换道、超车、并线等一系列决策中也表现良好。



![image-20200205113946025](/home/lichunhong/.config/Typora/typora-user-images/image-20200205113946025.png)



![image-20200205114009520](/home/lichunhong/.config/Typora/typora-user-images/image-20200205114009520.png)

![image-20200205114054116](/home/lichunhong/.config/Typora/typora-user-images/image-20200205114054116.png)









### 2009-Differentially Constrained Mobile Robot Motion Planning in State Lattices







lattice起源文章？









# 基于优化的运动规划



## 2018-Baidu Apollo EM Motion Planner

摘要-本文介绍了一个基于百度阿波罗（开源）自主驾驶平台的实时运动规划系统。开发的系统旨在解决工业4级运动规划问题，同时考虑安全性、舒适性和可扩展性。该系统以分层的方式覆盖多车道和单车道的自主驾驶：（1）系统顶层是一种多车道策略，通过比较并行计算的车道水平轨迹来处理车道变换情况。（2） 在车道级轨迹生成器中，基于Frenet框架迭代求解路径和速度优化问题。（3） 针对路径和速度优化问题，提出了动态规划和基于样条函数的二次规划相结合的方法，构造了一个可扩展且易于调整的框架，同时处理交通规则、障碍物决策和平滑度。该规划方法可扩展到高速公路和低速城市驾驶场景。我们还通过场景说明和道路测试结果演示了该算法。

### I. INTRODUCTION

**A. Multilane Strategy**



**B. Path-Speed Iterative Algorithm**



**C. Decisions and Traffic Regulations**

针对4级自主驾驶，决策模块应包括可扩展性和可行性。可伸缩性是场景表达能力（即可以解释的自主驾驶案例）。当考虑几十个障碍物时，决策行为很难用有限的自我汽车状态集来精确描述。对于可行性，我们的意思是，生成的决策应包括一个可行区域，在该可行区域内，ego车可以在动态限制内进行机动。然而，手动调整和基于模型的决策都不能生成无碰撞轨迹来验证其可行性。

在EM-planner的决策步骤中，我们以不同的方式描述行为。首先，用一个粗糙可行的轨迹来描述汽车的自我运动意图。然后，用此轨迹测量障碍物之间的相互作用。即使场景变得更加复杂，这种基于轨迹的可行决策也是可伸缩的。其次，规划器还将根据轨迹生成一个凸可行空间来平滑样条曲线参数。基于二次规划的平滑样条曲线解算器可以用来生成更平滑的路径和速度剖面。这保证了一个可行和顺利的解决方案。





### II. EM PLANNER FRAMEWORK WITH MULTILANE STRATEGY



![image-20200207185419069](/home/lichunhong/.config/Typora/typora-user-images/image-20200207185419069.png)



### III. EM PLANNER AT LANE LEVEL

![image-20200207185638460](/home/lichunhong/.config/Typora/typora-user-images/image-20200207185638460.png)

在第一个E-step，会将动静态障碍物投影到Frenet坐标系下，并且只考虑低速车辆和到来的障碍物，对于高速物体，EM Planner基于安全原因更倾向于变道。在第二个E-step，会考虑高速、低速车辆以及到来的障碍物。

两个M-step，首先使用DP在非凸的空间中生成粗糙的解，接下来使用QP进行凸优化求出平滑的解。



**A. SL and ST Mapping (E-step)**



SL映射

![image-20200207190809039](/home/lichunhong/.config/Typora/typora-user-images/image-20200207190809039.png)



ST映射

![image-20200207190834522](/home/lichunhong/.config/Typora/typora-user-images/image-20200207190834522.png)



**B. M-Step DP Path**

![image-20200207191432352](/home/lichunhong/.config/Typora/typora-user-images/image-20200207191432352.png)

![image-20200207191515221](/home/lichunhong/.config/Typora/typora-user-images/image-20200207191515221.png)



晶格采样是基于Frenet框架的。如图7所示，首先在ego车辆之前对多行点进行采样。不同行之间的点用五次多项式边光滑连接。点行之间的间隔距离取决于速度、道路结构、车道变换等。该框架允许根据应用程序场景自定义采样策略。例如，换道可能需要比当前车道行驶更长的采样间隔。此外，出于安全考虑，采样距离将至少覆盖8秒或200米。

在lattice轨迹构造之后，通过代价函数的求和来评价图的每一条边。我们使用SL投影、交通规则和车辆动力学的信息来构建函数。总边缘成本函数是平滑度、避障和车道成本函数的线性组合。

这里Cost有三个组成部分：平滑Cost、离障碍物距离Cost、离偏导线偏差Cost

![image-20200207193047535](/home/lichunhong/.config/Typora/typora-user-images/image-20200207193047535.png)

其中，

![image-20200207193109627](/home/lichunhong/.config/Typora/typora-user-images/image-20200207193109627.png)

![image-20200207193202141](/home/lichunhong/.config/Typora/typora-user-images/image-20200207193202141.png)

![image-20200207193221870](/home/lichunhong/.config/Typora/typora-user-images/image-20200207193221870.png)



Nudge Decision包括nudge，yield和overtake，用来生成convex hull来进行QP的spline优化。



**C. M-Step Spline QP Path**





![image-20200207193710133](/home/lichunhong/.config/Typora/typora-user-images/image-20200207193710133.png)





![image-20200207193818227](/home/lichunhong/.config/Typora/typora-user-images/image-20200207193818227.png)

QP的Cost相对简单些，就是负责平滑的路径一阶二阶三阶倒数还有DP结果与引导线的偏差。

**D. M-Step DP Speed Optimizer**

![image-20200207194405850](/home/lichunhong/.config/Typora/typora-user-images/image-20200207194405850.png)

![image-20200207194859303](/home/lichunhong/.config/Typora/typora-user-images/image-20200207194859303.png)

**E. M-Step QP Speed Optimizer**

![image-20200207195049570](/home/lichunhong/.config/Typora/typora-user-images/image-20200207195049570.png)

![image-20200207195107274](/home/lichunhong/.config/Typora/typora-user-images/image-20200207195107274.png)

**G. Notes on Non-convex Optimization With DP and QP**



### IV. CASE STUDY











参考：

[1] https://blog.csdn.net/yuxuan20062007/article/details/83629595



## Optimization-Based Collision Avoidance

摘要：利用凸优化的强对偶性，提出了一种将不可微碰撞避免约束转化为光滑非线性约束的新方法。我们关注的是一个控制对象，其目标是在n维空间中移动时避开障碍物。所提出的重构不引入近似，并且适用于一般的障碍物和受控对象，它们可以表示为凸集的并集。我们将我们的结果与符号距离的概念联系起来，符号距离在传统的轨迹生成算法中得到了广泛的应用。我们的方法可以应用于一般的导航和轨迹规划任务，并且平滑特性允许使用通用的基于梯度和Hessian的优化算法。最后，在无法避免碰撞的情况下，我们的框架允许我们找到“至少有吸引力”的轨迹，以穿透力来衡量。我们证明了我们的框架在四直升机导航和自动泊车问题上的有效性，并且我们的数值实验表明，所提出的方法能够在紧环境下实现基于实时优化的轨迹规划问题。我们实现的源代码见https://github.com/XiaojingGeorgeZhang/OBCA。

## 

## Integrated Online Trajectory Planning and Optimization in Distinctive Topologies

这篇文章**最为**详细地说明了TEB实现。

参考：

[1] http://www.pianshen.com/article/4783688865/

## Elastic Bands: Connecting Path Planning and Control

弹性带（EB）起源文章





## Kinodynamic Trajectory Optimization and Control for Car-Like Robots

这篇文章介绍**较为**详细地说明了TEB实现。





## Efficient Trajectory Optimization using a Sparse Model

**时间弹性带算法使用g2o框架求解**

摘要-“时间弹性带(TEB)”方法通过随后修改由全局规划器生成的初始轨迹来优化机器人轨迹。轨迹优化所考虑的目标包括但不限于总路径长度、轨迹执行时间、与障碍物的分离、通过中间路径点以及满足机器人的动力学、运动学和几何约束。TEB明确地考虑了运动的时空方面的动态约束，如有限的机器人速度和加速度。轨迹规划实时运行，使得TEB能够应对动态障碍物和运动约束。将“TEB问题”描述为一个尺度化的多目标优化问题。大多数目标是局部的，只与一小部分参数有关，因为它们只依赖于几个连续的机器人状态。这种局部结构产生一个稀疏的系统矩阵，从而允许使用快速有效的优化技术，如开源框架“g2o”来解决TEB问题。g2o稀疏系统解算器已成功地应用于VSLAM问题。本文描述了g2o框架在TEB轨迹修正中的应用和适应性。仿真和实际机器人实验结果表明，该方法具有良好的鲁棒性和计算效率。



### II. TIMED ELASTIC BAND

#### A. Definition of Timed Elastic Band (TEB)



<img src="./imags/image-20200119165353121.png" alt="image-20200119165353121" style="zoom:80%;" />



#### B. Problem representation as a Hyper-Graph



![image-20200119173454890](./imags/image-20200119173454890.png)

#### C. Control flow

<img src="./imags/image-20200119173521625.png" alt="image-20200119173521625" style="zoom:80%;" />



## G2o: A general framework for graph optimization

摘要-机器人学和计算机视觉中的许多常见问题，包括各种类型的同时定位和映射（SLAM）或束平差（BA），可以**用图形表示的误差函数的最小二乘优化来表达**。本文描述了这些问题的一般结构，并提出了**G2O，一个开源的C++框架**，用于**优化基于图的非线性误差函数**。我们的系统被设计成很容易扩展到各种各样的问题，一个新的问题通常可以在几行代码中指定。当前的实现为SLAM和BA的几种变体提供了解决方案。我们提供了对大量真实世界和模拟数据集的评估。结果表明，虽然g2o是通用的，但它的性能可以与针对特定问题的现有方法的实现相媲美。





g2o的本质：g2o是一个算法集的C++实现，而并不是在算法理论上的创新，即根据前人求解非线性最小二乘的理论，根据具体的问题，选用最合适的算法。

它是一个平台，你可以加入你自己的线性方程求解器，编写自己的优化目标函数，确定更新的方式。g2o的作者说Guassian-Newton和Levenberg-Marquardt方法比较naive，但是g2o的本质就是这些算法的实现。事实上，g2o iSAM SPA和 sSPA等非线性优化算法只是在非线性问题线性化时处理得不一样，在线性化后要求解线性方程都是利用了已有的linear solver库来求解，如 CSparse CHOLMOD PCG等，他们都需要依靠Eigen这个线性代数库。

g2o的用途：很多机器人的应用如SLAM（同步定位与制图）还有计算机视觉中的光束优化（bundle adjustment 都会涉及到最小化非线性误差函数的问题。这类应用中，**非线性误差函数可以用图(graph)的形式来表征**。整个问题的求解就是要找到最符合观测量的相机参数或机器人状态。

参考：

[1] https://blog.csdn.net/zhongjin616/article/details/15498779





# 组合运动规划（路线图）

　











# 路径跟踪





## Geometric Path Tracking Algorithm for Autonomous Driving in Pedestrian Environment

摘要：本文提出了一种用于自动驾驶的纯跟踪路径跟踪算法的替代公式。目前的方法有偷工减料的倾向，因此导致路径跟踪精度差。该方法**不仅考虑了被跟踪点的相对位置，而且还考虑了被跟踪点的路径方向**。根据车辆运动方程设计了转向控制律。该算法的有效性通过在无人驾驶的高尔夫球车上实现，并在步行环境下进行了测试。实验结果表明，新算法在不增加额外计算量的情况下，使同一给定预设路径的均方根（RMS）交叉跟踪误差降低46%，且**保持了原纯跟踪控制器的无抖振特性**。



改进了PPC算法： 减少过弯道时切角和超调，保持无抖振特性。

### III. PURE PURSUIT PATH TRACKING

#### B. Pure Pursuit Algorithm

![image-20200121170515047](./imags/image-20200121170515047.png)

图中，$Ｌ$为车辆轴距，$L_{fw}$为lookahead距离。$L_{fw}=kv(t)\in{[L_{min},L_{max}]}$



根据Fig. 4，三角形定理有：

$$\frac{L_{fw}}{sin(2\eta)}=\frac{R}{sin(\frac{\pi}{2}-\eta)}$$

$$\frac{L_{fw}}{２sin(\eta)cos(\eta)}=\frac{R}{cos(\eta)}$$

$$\frac{L_{fw}}{sin(\eta)}=2R$$

那么，曲率$\kappa=\frac{1}{R}=\frac{2sin(\eta)}{L_{fw}}$



（１）以$(v,\omega)$控制的底盘

很多移动机器人，较为代表的是差速轮底盘的移动机器人，车辆的控制指令通常为$(v,\omega)$，而$\omega=v(t)\kappa$

所以，计算出转角$\eta$，根据当前车辆速度$v(t)$，便可求出需要的角速度$\omega$，下发控制指令$(v,\omega)$即可。

（２）以$(v,\delta)$控制的底盘

较为代表的是乘用车为代表的阿卡曼模型和全驱动的双阿克曼模型，车辆控制指令为$(v,\delta)$

![image-20200121170455903](./imags/image-20200121170455903.png)

对于阿克曼模型

$$tan(\delta)=\frac{L}{R}$$

$$\delta=tan^{-1}(\kappa L)$$

$$\delta(t)=tan^{-1}(\frac{2Lsin(\eta(t))}{L_{fw}})$$

(2)对于双阿克曼模型

$$tan(\delta)=\frac{L}{2R}$$

$$\delta=tan^{-1}(\frac{\kappa L}{2})$$

$$\delta(t)=tan^{-1}(\frac{Lsin(\eta(t))}{L_{fw}})$$





### IV. MODIFIED PURE PURSUIT PATH TRACKING



![image-20200122103546778](./imags/image-20200122103546778.png)



改进的PPC算法考虑了被跟踪点$(x_{p},y_{p})$的方向（$i$点的方向由$i+1$点的连线方向确定）。

![20200122103155562](./imags/image-20200122103155562.png)

![20200122103155562](./imags/image-20200122103941301.png)



当考虑了被跟踪点的方向后，跟踪时会产生垂直偏移$d$，这会导致跟踪固定曲率的路径时产生稳态误差，因此需要补偿$d$。



![image-20200122104528388](./imags/image-20200122104528388.png)

补偿方式：将跟踪点的位置$(x_{p},y_{p},\theta_{p})$，沿着$(\theta_{p}+\frac{\pi}{2})$方向，偏移$-d$的距离作为补偿后的跟踪位置，计算公式如下：
![image-20200122111240791](./imags/image-20200122111240791.png)

对应的转角用纯跟踪算法计算即可。





# 盲区感知



## Autonomous Predictive Driving for Blind Intersections



![image-20200121112430633](./imags/image-20200121112430633.png)





# Autoware实现

## Open Source Integrated Planner for Autonomous Navigation in Highly Dynamic Environments

这篇文章主要介绍Autoware的系统架构和部分实现细节。



### 5. Local Planner

#### 5.1. Roll-Out Generation

**样条曲线：**

Autoware局部路径规划所使用的样条曲线，分为三段：car tip margin， roll-in margin，  roll-out section，使得转角平滑。

**样条插值：**

很多样条插值方法对输入噪声敏感，如当输入点过于紧密时，三次样条插值方法会产生严重震荡。Autoware 使用分段插值（piece wise interpolation）和共轭梯度（conjugate gradient）平滑的方法生成路径点。

![image-20200117134814993](./imags/image-20200117134814993.png)



> 共轭梯度（conjugate gradient）平滑方法
>
> TODO



<img src="./imags/image-20200117141115384.png" alt="image-20200117141115384" style="zoom:80%;" />

#### 5.2. Cost Calculation



代价指标：priority cost, collision cost and transition cost

**障碍物表示**

障碍物表示：Ｂounding Boxes 、点云簇

优缺点：Ｂounding Boxes精度低，障碍物检测需要计算性能高；点云簇正好相反。

Ａutoware改进了障碍物的点云簇表示，最多采样16个点（点数可配置）就可以表示一个物体。



##### 5.2.1. Center Cost

与中心参考线距离的代价

##### 5.2.2. Transition Cost

各条roll-outs与当前选择的路径的垂直距离

##### 5.2.3. Collision Cost

分两段：

第一段是car tip margin＋roll-in margin连接的样条

碰撞检测使用“point inside a circle”，以路径点为圆心，车宽的一半＋安全距离为半径，看障碍物轮廓点是否在圆内即可。

第一段是 roll-out section样条

由于样条是平行的，可方便的计算是否碰撞



### 6. Behavior Generation Using State Machine

Behavior states transition conditions.

![image-20200117151720773](./imags/image-20200117151720773.png)



### 几点疑问

１、动态障碍物的碰撞检测只检查空间上的碰撞，没检查时间上的碰撞？

２、采样的轨迹没有速度信息，速度是如何给定的？







