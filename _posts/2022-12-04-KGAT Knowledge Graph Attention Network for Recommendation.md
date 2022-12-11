---
title: "KGAT Knowledge Graph Aention Network for Recommendation"
excerpt: '스프링 부트 입문 03, 04'
categories:
    - Spring Boot

tag:
    - Spring Boot 
    - JAVA

author_profile: true    #작성자 프로필 출력 여부

last_modified_at: 2022-12-11T15:46:00+09:00

toc: true   #Table Of Contents 목차 

toc_sticky: true

---

## KGAT Knowledge Graph Aention Network for Recommendation

<br/>

#### 목차

- [INTRODUCTION](### INTRODUCTION)
    - Collaborative Filtering/SL model 의 한계
    - Knowledge Graph Attention Network (KGAT)
- [METHODOLOGY](### METHODOLOGY)
    - embedding layer
    - Attentive Embedding Propagation Layers
    - Model Prediction
- [EXPERIMENTS](### EXPERIMENTS)
  

<br/>


### 1.INTRODUCTION
---

#### Collaborative Filtering/SL model 의 한계

<img src="https://jaeeun49.github.io/images/KGAT/CF.png"> 

CF 방식은 여러 사용자들로부터 얻은 정보를 이용해 사용자들의 관심사들을 예측하게 해주는 방법입니다. 하지만 CF 방식은 side information(아이템 속성, 유저 속성 등)을 모델링하는 것에 제약이 있습니다. 그래서 side information을 반영하기 위해 유저와 아이템의 속성을 실수 형태로 표현하고 supervised learning으로 학습하는 SL 모델이 널리 사용되어 왔고 대표적인 SL 모델로는 factorization machine(FM), NFM (neural FM), Wide&Deep, xDeepFM 모델이 있습니다.

<br/>

<img src="https://jaeeun49.github.io/images/KGAT/FM.png">

factorization machine의 feature vector x를 보여주고 있습니다. user의 one-hot vector, item의 one-hot vector, user가 평가한 다른 item의 평점, 해당 item을 평가하기 바로 직전 평가한 item에 대한 정보 등 side information을 반영해 모델링 하여 높은 성능을 보여주었습니다. 하지만 이러한 SL 모델은 각각의 interation을 독립적으로 보아 연결관계를 고려하지 않습니다. 

<br/>

<img src="https://jaeeun49.github.io/images/KGAT/figure1.png">

논문에서 collaborative knowledge graph의 예시로 보여주고 있는 그림이며 이 그림을 통해 CF와 SL이 동작하는 원리를 아래와 같이 간단히 말해주고 있습니다.그래서 SL 모델의 경우 yellow circle, grey circle 같이 high-order connectivity(the long-range connectivities)을 고려하지 못합니다.

- CF) user1이 item1을 보았기 때문에 item1을 본 다른 유저 user4와 user5가 본 다른 item에 집중합니다.
- SL) item1의 속성 e1과 비슷한 item인 i2에 집중합니다.


<img src="https://jaeeun49.github.io/images/KGAT/circle.png"> 

다음이 각각 yellow circle, grey circle을 표현하고 있으며 이러한 high-order information을 고려하는 것이 추천에 핵심이지만 이것은 다음과 같은 이유로 매우 어려운 문제입니다.
1. high-order relations을 가지는 노드들은 target user가 증가할 수록 극적으로 사이즈가 커져 모델에 과부하를 줄 수 있습니다.
2. high-order relations은 예측에 동일하게 기여하지 않기 때문에 모델은 각 연결에 대한 가중치를 부여해야 합니다.

<br/>

<br/>

#### Knowledge Graph Attention Network (KGAT)

그래서 저자는 이러한 high-order relation을 모델링하는데 있어 어려움을 해결하기 위해 two designs으로 구성된 Knowledge Graph Attention Network (KGAT)을 제안하고 있습니다.

저자는 이 논문의 기여를 다음과 같이 요약하고 있습니다.
- 더 좋은 추천을 위해 knowledge graph에서의 high-order relations을 모델링에 대한 중요성을 강조한다
- graph neural network 구조의 end-to-end 방법으로 high-order relations을 모델링할 수 있는 KGAT를 개발했다.
- 세가지 benchmarks으로 실험을 하였고 KGAT의 효율성과 high-order relations을 고려할 수 있는 능력을 입증했다.

<br/>
<br/>



### 2.METHODOLOGY
---

<img src="https://jaeeun49.github.io/images/KGAT/figure2.png">

KGAT 모델의 framework를 보여주고 있고 3가지 요소로 구성되어 있습니다.

1. embedding layer - CKG 구조로 각각의 노드를 vector로 변환
2. attentive embedding propagation layers - 이웃노드의 embedding을 반복적으로 propagate하여 노드 각각을 update 
3. prediction layer

<br/>

#### embedding layer

우선 Knowledge graph의 Entity와 Relation 을 벡터로 표현하는 것이 필요합니다. 저자들은 이 과정에서 TransR 방법을 사용하고 있는데, 벡터 공간에 표현된 트리플 <Head, Relation, Tail>을 더하기, 곱하기 등의 연산을 사용해 표현하는 방법 중 하나입니다. 

<img src="https://jaeeun49.github.io/images/KGAT/transE.png">
- 출처: https://wigo.tistory.com/entry/KG3-Translation-Model-for-KC

tansR을 보기 전에 TransE를 먼저 보면 head entity 와 relation의 벡터를 더하면 tail entity의 벡터가 된다는 접근 방식을 사용하고 있습니다. 즉, 위의 figure2 예시로 적용해 보면 u1 vector + r1 vector = i1 vector 가 됩니다. 

<img src="https://jaeeun49.github.io/images/KGAT/tansR.png">

<img src="https://jaeeun49.github.io/images/KGAT/transR2.png">

eh,et,er은 각각 h(head),t(tail),r(relation)의 임베딩 벡터를 표현하는 것이고 transR은 eh,et을 relation 공간으로 투영해 Entity와 Relation을 동일한 차원에 표현되는 것을 개선하였습니다. 이는 TransE의 문제점을 개선하기 위함인데 TransE의 경우 u1 vector + r1 vector = u4 vector + r1 vector = u5 vector + r1 vector = i1 vector 가 성립되지만 다른 relation에 적용한다면 문제가 생길 수 있습니다. 그래서 위 식에서 보이는 Wr는 relation r에 대한 transformation matrix이 되어 eh와 et를 각 relation 공간으로 투영해줍니다. 

그리고 embedding layer 학습 방법은 다음과 같은 pairwise ranking loss를 이용합니다. g(h,r,t)이 실제 정답 triple을 표현한다면 g(h,r,t0)은 하나의 entity를 틀린 값으로 대체해 그 차이를 이용해 학습합니다.

<img src="https://jaeeun49.github.io/images/KGAT/trainembedding.png">

<br/>

<br/>

#### Attentive Embedding Propagation Layers

두번째 레이어가 graph attention network를 이용해 연결성의 중요도를 반영하기 위한 attentive weights를 만들어 주는 부분입니다. 논문에서는 일단 single layer로 먼저 설명하고 세가지 요소로 나누어 설명하고 있습니다. 

1. information propagation
2. knowledge-aware attention
3. information aggregation

그리고 논문에서는 위의 3가지 요소를 head가 i3일때의 예시를 들어 그림으로 표현해주고 있습니다.

<img src="https://jaeeun49.github.io/images/KGAT/example1.png">


<img src="https://jaeeun49.github.io/images/KGAT/example2.png">

위의 예시를 더 보편적으로 설명하면 entity h에 대해 h가 연결된 triplets 집합인 Nh {(h,r,t)}들을 정의하고 h의 여러 연결관계의 선형결합(이를 ego-network라고 부르고 있습니다)을 통해 h에 연결된 t의 정보를 전달하는 Information Propagation이 수행됩니다. 그리고 이때 relation r공간에서의 eh와 et의 거리를 반영하는 attention score를 계산해 가까운 entity들의 경우 더 많은 정보가 전달되게 하는 Knowledge-aware Attention가 수행됩니다. 이를 통해 어느 정보에 더 집중해야 하는지를 전달할 수 있습니다. 이후 가장 마지막 단계는 Information Aggregation으로 entity representation eh와 ego-network representations eNh를 결합하는 과정을 거치게 됩니다. 결합함수로 여러개를 사용할 수 있고 위의 예시는 합과 곱 두 종류를 사용하는 Bi-Interaction Aggregator를 보여주고 있습니다. 

<br/>

#### Model Prediction

<br/>

<br/>


### 3.EXPERIMENTS
---

<br/>

<br/>
