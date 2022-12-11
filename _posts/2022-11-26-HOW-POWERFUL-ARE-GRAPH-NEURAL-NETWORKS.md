---
title: "[Paper Review] HOW POWERFUL ARE GRAPH NEURAL NETWORKS."
excerpt: 'GRAPH NEURAL NETWORKS'
categories:
    - Paper Review

tag:
    - GNN

author_profile: true    #작성자 프로필 출력 여부

last_modified_at: 2022-12-11T16:23:00+09:00

toc: true   #Table Of Contents 목차 

toc_sticky: true

---


#### 목차

1.[미리 알면 좋을 내용](#### 미리 알아두기)  
2.[ABSTRACT](### ABSTRACT)  
3.[INTRODUCTION](### INTRODUCTION)  
4.[PRELIMINARIES](### PRELIMINARIES)  
5.[THEORETICAL FRAMEWORK: OVERVIEW](### THEORETICAL FRAMEWORK: OVERVIEW)  
6.[BUILDING POWERFUL GRAPH NEURAL NETWORKS](### BUILDING POWERFUL GRAPH NEURAL NETWORKS)  
7.[LESS POWERFUL BUT STILL INTERESTING GNNS](### LESS POWERFUL BUT STILL INTERESTING GNNS)    
8.[EXPERIMENTS](### EXPERIMENTS)  

<br/>

#### 미리 알아두기
- representation learning
- Weisfeiler-Lehman graph isomorphism test
- injective aggregation
- universal approximation theorem


<br/>


### ABSTRACT
---

GNN은 그래프의 representation learning 에 대해 효율적인 프레임워크이다. GNN은 neighborhood aggregation를 따르며, 한 노드의 표현벡터는 주변 노드들의 표현 벡터를 반복적으로 aggregating, transforming 해 계산되어진다. 많은 GNN variant들이 제안되었고, node and graph classification tasks에서 최고의 성능을 보여주었다. 하지만 GNN의 혁신적인  graph representation learning에도 불구하고 representational 특징이나 한계에 대한 이해는 제한되어 있다. 여기서 우리는 다른 그래프 구조를 알아내는 GNN의 power를 분석하는데 있어 이론적인 framework를 제시한다. 여러 GNN 중에서도 가장 expressive한 간단한 구조를 발전시켰으며 Weisfeiler-Lehman graph isomorphism test 만큼이나 powerful하다. 우리는 많은 graph classification benchmarks에서 우리의 이론적 발견을 증명하였고, 이 모델이 최고의 성능을 보여주고 있음을 증명하였다. 

   
<br/>
 
### INTRODUCTION
---

최근 그래프의 representation learning의 접근으로 GNN이 큰 관심을 받고 있다. GNN은 여러 iteration을 거쳐 각각의 노드가 이웃노드의feature vectors를 수집하고 새로운 feature vectors로 업데이트 하는 recursive neighborhood aggregation (or message passing) schema를 따른다. 그리고 이러한 방법으로 노드의 k-hop neighborhood 내에서의 구조적인 정보를 파악한다. 그래프의 모든 노드에 대해 representation vectors를 Pooling 함으로써, 예를 들어 더하는 방법으로 전체적인 그래프의 representation을 얻을 수 있다. 


이웃 노드를 수집하는 방법과 graph-level pooling schemes가 다른 여러 GNN variant가 제안되었다. 분명 이처럼 다양한 GNN이 node classification, link prediction, and graph classification과 같이 여러 과제에서 좋은 성능을 보여주었지만 대부분 직관적이고 휴리스틱적이며, 실험기반의 시행 착오를 통해 이루어진다. GNN의 특징과 한계에 대해 이론적인 이해가 부족하며, GNN의 representational 능력에 대한 공식적인 분석이 부족하다.


그래서 우리는, GNN의 representational power를 분석할 수 있는 이론적인 프레임워크를 제시한다. 이 프레임워크는 GNN과 Weisfeiler-Lehman (WL) graph isomorphism test이 가까운 연결 관계를 가진다는 점에서 영감을 받았다. Weisfeiler-Lehman (WL) graph isomorphism test는 다양한 종류의 그래프를 구별해 내는데 있어 강력한 테스트로 알려져 있으며 GNN과 비슷하게 network neighbors의 feature vectors를 수집하면서 노드의 feature vector를 업데이트 한다. WL test를 강력히 만들어 주는 것은 injective aggregation 업데이트이며 이것이 다른 이웃 노드들을 다른 feature vectors로 매핑시켜준다. 우리의 핵심은 만약 GNN의 aggregation scheme이 매우 model injective functions이라면 WL test처럼 매우 강력히 구별할 수 있는 능력을 가지게 될 것이라는 점이다. 


위의 내용을 수학적으로 증명하기 위해, 우리의 프레임워크는 이웃 노드의 feature vector들을 multiset(동일한 요소들이 존재 할 수 있는)으로 표현한다. 그럼 GNN의 neighbor aggregation 즉 이웃노드들을 수집하는 것은 이 multiset의 aggregation function으로 생각할 수 있다. 우리는 이 multiset functions에 대해 여러 변형들을 연구하고 이들의 discriminative power를 묘사한다. 그 multiset function이 더 잘 구별해낼 수록 더 GNN은 powerful한 representational power를 가진 것이다.


주요 결과는 다음과 같이 요약된다.
- GNN이 그래프 구조를 구별해 내는데 있어 적어도 WL test 만큼 강력하다.
- GNN의 결과가 WL test 만큼이나 강력함에 있어 neighbor aggregation이나 graph readout functions에 대해 조건을 주었다.
- GCN (Kipf & Welling, 2017)나 GraphSAGE (Hamilton et al., 2017a) 처럼 인기있는 GNN들이 구별해 내지 못하는 그래프 구조와 파악할 수 있는 그래프 구조를 확인했다.
- 간단한 neural architecture, Graph Isomorphism Network (GIN)을 개발했고, 이것의  discriminative/representational power가 WL test 만큼이나 강력하다는 것을 보여준다.


<br/>

<br/>

### PRELIMINARIES
---

#### 표기법 정리/ 용어 설명
- Let G = (V, E) : node feature vectors Xv (v ∈ V)를 가지는 그래프
- Node classification : 각각의 노드(v ∈ V)는 label yv를 가질 것이고 결국은 노드(v)의 representation vector인 hv를 학습해서 yv = f(hv) 을 통해 노드의 정답을 예측하는 task를 말함
- Graph classification : 그래프 집합 {G1, ..., GN } ⊆ G 와 그들의 label인 {y1, ..., yN } ⊆ Y이 주어지면 그래프의 representation vector인 hG을 학습시켜 yG = g(hG)을 통해 label을 예측한 task


#### Graph Neural Networks
<img src="https://user-images.githubusercontent.com/76995436/204095140-978492b1-e37d-4839-b95b-25d83fc1dcc1.PNG" >
- h(k)v : k번째 iteration을 돌고 난 후 node v의 feature vector
- N (v) : 노드 v에 인접한 노드들의 집합

GNN은 neighborhood aggregation을 따르는데, 이는 각 노드에 인접한 이웃 노드들의 feature vector을 수집함으로써 노드의 representation을 update하는 방식이다. 이러한 aggregation을 k번 반복하고 난 후, 노드의  representation이 k-hop network neighborhood내에서의 구조적 정보를 파악한다. 위의 식에서 알 수 있듯이 GNN에 있어 AGGREGATE(k)(·) 와 COMBINE(k)(·)의 방식이 중요하고 다양한 구조들이 제안되었다. 

이를 더 순차적으로 정리해보면
1. neighborhood의 feature vector들을 AGGREGATE
2. 1번의 결과로 얻은 a(k)v와 현재 feature vector를 combine해 feature vector update


#### Weisfeiler-Lehman test.

The graph isomorphism problem은 두 그래프가 위상적으로 동일한지를 묻는다. Weisfeiler-Lehman (WL) test of graph isomorphism (Weisfeiler & Lehman, 1968)는 다양한 종류의 그래프를 구별하는 아주 효울적인 테스트이다. 이것의 1차원적인 모양인 “naïve vertex refinement”는 GNN에 있어 neighbor aggregation와 유사하다. WL test는 이웃 노드들의 label을 수집하고 그렇게 수집한 labels을 새로운 labels로 매핑한다. 이 알고리즘은 어느 시점에서 두 그래프 노드들의 label들이 다르면 non-isomorphic하다고 판단한다. 


<br/>

<br/>
 
### THEORETICAL FRAMEWORK: OVERVIEW
---

<img src="https://jaeeun49.github.io/images/How_Powerful_are_Graph_Neural_Networks/Figure1.png"> 

GNN은 노드 주변의 다른 노드들의 네트워크 구조와 특징을 파악하기 위해 반복적으로 각 노드의 feature vector를 업데이트 한다. 논문에서는 내내 node input features가 유한하다고 가정하고 있다. 그래서 간단히 각각의 feature vector에게 하나의 고유한 label을 줄 수 있으며 이웃노드의 feature vectors 집합은 multiset으로 표현한다.



<br/>

<br/>
   
### BUILDING POWERFUL GRAPH NEURAL NETWORKS
---

이상적으로 가장 powerful한 GNN은 서로 다른 그래프들을 embedding space에서 다른 공간에 매핑시킴으로써 그래프 구조를 구별할 수 있다. 즉 우리는 동일 구조의 그래프들은 같은 representation을 갖도록 다른 구조를 가지는 그래프들은 다른 representation을 갖기를 바란다. 우리의 분석에서 우리는 GNN의 representational 능력을 잘 알려지고 강력한 Weisfeiler-Lehman (WL) graph isomorphism를 통해 알아내고자 한다. 

> <img src="https://jaeeun49.github.io/images/How_Powerful_are_Graph_Neural_Networks/Lemma 2.png"> 

어느 aggregation 함수를 갖는 GNN도 서로 다른 그래프를 구별하는데 있어 WL test 만큼이나 powerful하다. 자연스럽게 따라오는 질문은 기존의 GNN들이 WL test 만큼이나 powerful하냐는 것이다. 그리고 Theorem 3에서 그렇다고 말하고 있다. 만약 neighbor aggregation과 graph-level readout functions이 injective하다면 GNN은 WL test 만큼이나 강력하다.

> <img src="https://jaeeun49.github.io/images/How_Powerful_are_Graph_Neural_Networks/Theorem 3.png"> 

Theorem 3은 다음 두 조건을 만족하면 GNN이 WL test가 다른 구조의 그래프를 다른 embedding으로 구별하는 것처럼 작용한다고 말한다.
- 셀수 있는 집합인 multiset에 적용되는 함수 f와 φ 가 injective 하다
- node features의 multiset에 적용되는 graph-level readout는 injective 하다.

> <img src="https://jaeeun49.github.io/images/How_Powerful_are_Graph_Neural_Networks/Lemma 4.png"> 

- input node features가 셀수 있는 집합으로 표현될 수 있는 경우만을 고려한다.

여기서 GNN이 서로 다른 그래프를 구별해 내는 것 외에 중요한 이점, 그래프 구조의 유사성을 파악할 수 있는 이점을 얘기하고 싶다. WL test에서 node feature vectors는 one-hot encoding이기 때문에 subtree들 끼리의 유사성은 파악할 수 없다. 반대로 GNN은 subtree를 낮은 차원으로 embed하는 것을 학습함으로써 WL test를 일반화한다. 그래서 GNN은 다른 구조를 구별해 낼 수 있을 뿐 아니라 비슷한 그래프 구조는 비슷한 embbedding으로 매핑시킬 수 있다. 



#### GRAPH ISOMORPHISM NETWORK (GIN)

우리는 간단한 구조인 Graph Isomorphism Network (GIN)을 개발했고 Theorem 3의 조건을 만족한다. 이 모델은 WL test를 일반화하여 여러 GNN들 사이에서도 매우 강한 discriminative 능력을 보여준다. 이웃노드를 수집하는데 있어 injective multiset functions을 모델링 하기 위해 우리는 neural networks을 사용해 multiset functions의 파라미터를 학습시키는 “deep multisets” 이론을 발전시켰다. 다음 lemma는 sum aggregators이 multiset에 대해 injective한 함수를 나타낼 수 있다고 말하고 있다.

> <img src="https://jaeeun49.github.io/images/How_Powerful_are_Graph_Neural_Networks/Lemma 5.png">

Lemma 5는 환경을 set에서 multiset으로 확장한다. multiset과 set의 가장 중요한 차이는 mean aggregator처럼 인기있는 injective set 함수가 injective multiset 함수는 아니라는 점이다. Lemma 5에서 보듯 보편적인 multiset functions를 모델링 하는 메커니즘으로 우리는 한 노드와 그 이웃노드의 multiset에 적용되는 universal functions을 나타낼 수 있는 aggregation schema를 생각할 수 있다. 그리고 다음의 corollary이 그러한 많은 aggregation schemes 중에서도 간단한 공식을 제공한다.

**다음 Corollary 6부터 MLP 수식까지 한번 더 보기**

> <img src="https://jaeeun49.github.io/images/How_Powerful_are_Graph_Neural_Networks/Corollary.png">

보편적 근사 정리(universal approximation theorem) 덕분에 다층 퍼셉트론(MLP)을 사용하여 Corollary 6에서 f와 θ를 모델링하고 학습할 수 있다 

<img src="https://jaeeun49.github.io/images/How_Powerful_are_Graph_Neural_Networks/Corollary6수식.png">



<br/>

<br/>
   
### LESS POWERFUL BUT STILL INTERESTING GNNS
---

다음으로 GCN과 GraphSAGE를 포함해 Theorem 3에서의 조건을 만족하지 않는 GNN을 연구할 것이다.
Eq. 4.1에서의 aggregator의 두가지 측면에 대해 ablation studies를 수행한다.
1. 1-layer perceptrons instead of MLPs
2. mean or max-pooling instead of the sum

이 두가지 변형이 놀라울 정도로 그래프에 혼란을 주며 WL test에 비해 less powerful하게 됨을 볼 것이다. 하지만 GCN처럼 mean aggregators을 가지는 모델들은 node classification tasks의 경우 잘 수행한다.

   
####  1-LAYER PERCEPTRONS ARE NOT SUFFICIENT

Lemma 5에서 f 함수는 multisets을 유일한 embedding으로 매핑시켜주는 기능을 한다. 이 함수는 universal approximation theorem에 의해 MLP로 파라미터 값이 학습될 수 있다. 하지만 많은 GNN들이 MLP대신  1-layer perceptron인 σ ◦ W (a linear mapping 이후 ReLU와 같이 비선형 활성화 함수을 통과하는) 을 사용한다. 이러한  1-layer mappings은  Generalized Linear Models의 예시이다. 그래서 우리는 1-layer perceptrons이 그래프 학습에도 충분한지를 이해하는 것이 흥미로웠다. Lemma 7은 1-layer perceptrons은 결코 network neighborhoods (multisets)을 구별해내지 못함을 나타낸다. 

> <img src="https://jaeeun49.github.io/images/How_Powerful_are_Graph_Neural_Networks/Lemma7.png"> 

- Lemma 7의 핵심 내용은 1-layer perceptrons은 선형 매핑과 매우 유사하게 동작할 수 있으므로 GNN 레이어는 이웃 기능에 대한 단순 합산으로 퇴화한다는 것이다. MLP를 사용하는 모델과 달리 1-layer perceptron은 바이어스 항이 있더라도 Universal approximation theorem이 성립되지 않는다. 결과적으로 1-layer perceptron이 어느정도 서로 다른 그래프를 다른 location으로 embedding 할 수 있더라도 이 결과는 구조적 유사성을 파악하기에 충분하지 않는다. 

   
#### STRUCTURES THAT CONFUSE MEAN AND MAX-POOLING

feature vector를 combine하는 부분을 sum에서 GCN이나 GraphSAGE에서 사용하는 mean이나 max-pooling으로 바꾸면 어떻게 될까? Mean and max-pooling aggregators은 multiset 함수이지만 injective하지 않다.


<img src="https://jaeeun49.github.io/images/How_Powerful_are_Graph_Neural_Networks/figure3.png">

- 같은 색의 노드는 같은 node features를 가진다.
- a의 경우 모든 노드가 같은 feature a를 가지고 있고 mean, max 어느 경우든 f(a)가 같다. 그래서 neighborhood aggregation를 수행하면 f(a)에 대해 mean 이나 maximum 값은 f(a)가 된다. 항상 같은 node representation를 가질 것이다. 하지만 sum aggregator라면 2 · f(a)와 3 · f(a) 의 값은 다르기 때문에 서로 다른 그래프를 구별할 수 있게 된다.
- b는 blue nodes인 v and v0의 이웃노드 maximum 값이 max (hg, hr)과 max (hg, hr, hr)을 산출함을 보여준다. 그리고 이것은 그래프 구조가 다름에도 불구하고 동일한 표현으로 축소된다. 
- c의 경우 1/2 * (hg + hr) =  1/4 * (hg + hg + hr + hr) 이기 때문에 mean, max 둘다 서로 다른 그래프를 구별하는 것에 실패했다.

   
####  REMARKS ON OTHER AGGREGATORS

여기서 다루지 않는 다른 neighbor aggregation 방식(weighted average via attention 이나 LSTM pooling 등)들도 있다. 우리는 우리의 이론적 프레임워크가 어떠한 aggregation을 가진 GNN에 대해 representaional power를 표현하기에 충분히 일반적임을 강조한다. 미래에 우리의 프레임워크를 또 다른 aggregation schemes를 분석하고 이해하기 위해 응요해 본다면 흥미로울 것 같다.


<br/>

<br/>
   
### EXPERIMENTS
---

<img src="https://jaeeun49.github.io/images/How_Powerful_are_Graph_Neural_Networks/Figure4.png">

<img src="https://jaeeun49.github.io/images/How_Powerful_are_Graph_Neural_Networks/Corollary6수식.png">

- Datasets: 9 graph classification benchmarks
    - 4 bioinformatics datasets
    - 5 social network datasets
- Models and configurations
    - state-of-the-art baselines
      * the WL subtree kernel
      * Diffusion convolutional neural networks (DCNN)
      * Anonymous Walk Embeddings (AWL)
    - GIN
      * GIN-eps that learns eps by gradient descent
      * GIN-0 that eps is fixed to 0
    - the less powerful GNN variants
      * summation을 mean, max-pooling, MLPs with 1-layer perceptrons로 바꿔줌
      * GraphSAGE (max–1-layer)
      * GCN (mean–1-layer)

<img src="https://jaeeun49.github.io/images/How_Powerful_are_Graph_Neural_Networks/table1.png">

- Figure 4를 보면 GIN-eps, GIN-0 둘다 training accuracy 가 거의 1 에 수렴
- Table 1은 각 모델이 9가지 dataset에 대해 10 folds로 cross-validation한 결과의 the average and standard deviation of validation accuracies를 보여주고 있음
-  MLPs를 가진 GNN들이 1-layer perceptrons의 GNN보다 더 높은 training accuracies를 가짐
-  sum aggregators이 mean and max-pooling aggregators보다 train set을 더 잘 학습하는 경향이 있음
-  모든 GNN 모델들은 WL subtree kernel 의 정확도보다 낮음
-  GIN-0은 9가지 모든 dataset에서 다른 GNN 변형에 비해 최고의 성능을 보여줌
-  GINs 끼리 비교해보면 아주 작은 차이지만 GIN-0이 GIN-eps보다 성능이 높음. 두 모델 모두 데이터를 잘 학습하지만 더 일반화 모델인 GIN-0이 GIN-eps와 비교해서 simplicity에 의해 더 잘 설명된다.
-  GIN은 상대적으로 학습시킬 그래프의 수가 많은 social network에서 빛을 보여줌


<br/>

<br/>
