# Attention Mechanisms Through the Lens of Numerical Methods
## — Approximation Methods and Alternative Formulations

### 组会报告（约1.5小时，覆盖论文 Section 1-5）

---

## 论文信息

- **标题**: Attention Mechanisms Through the Lens of Numerical Methods: Approximation Methods and Alternative Formulations
- **类型**: Preprint (arXiv:2604.01757v1, 2026年4月2日)
- **背景**: 源自 IPAM (Institute for Pure and Applied Mathematics) 的 "Randomized Numerical Linear Algebra (RNLA) 2025" 研讨会

---

## 作者详细信息

  

### 项目组长 (†)

  

| 作者 | 单位 | 研究方向 | 链接 |

|------|------|---------|------|

| **Laura Grigori** † | EPFL & Paul Scherrer Institute, 瑞士 | 数值线性代数、通信避免算法、高性能计算 | [EPFL主页](https://www.epfl.ch/labs/hpnalgs/hpnalgs-chair-of-high-performance-numerical-algorithms-and-simulations/prof-laura-grigori/) · [Google Scholar](https://scholar.google.com/citations?user=BST6b8AAAAAJ&hl=en&oi=ao) · [Wikipedia](https://en.wikipedia.org/wiki/Laura_Grigori) · [ORCID](https://orcid.org/0000-0002-5880-1076) |

  

> Laura Grigori 是法裔罗马尼亚应用数学家，EPFL 应用与计算数学教授，PSI 科学计算中心成员。曾任 INRIA 研究主任，2014-2016 及 2022-2023 年在 UC Berkeley 访问。以通信避免算法（communication-avoiding algorithms）闻名。

  

### 项目共同组长 (*)

  

| 作者 | 单位 | 研究方向 | 链接 |

|------|------|---------|------|

| **Michel Fabrice Serret** * | Paul Scherrer Institute, 瑞士 | Transformer 与注意力机制的数学分析 | [LJLL主页](https://www.ljll.fr/en/membre/serret-michel-fabrice/) · 邮箱: michel.serret@psi.ch |

| **Alice Cortinovis** * | University of Pisa, 意大利 (INdAM/GNCS 成员) | 数值线性代数、随机化算法、矩阵近似 | [个人主页](https://sites.google.com/view/alicecortinovis/home) · [Google Scholar](https://scholar.google.com/citations?user=ZdVDls8AAAAJ)  |

  

> Alice Cortinovis 曾任 Stanford 大学 Szegö 助理教授，现为 Pisa 大学计算机系 tenure-track 研究员。

  

### 其他作者

  

| 作者 | 单位 | 研究方向 | 链接 |

|------|------|---------|------|

| **Yijun Dong** | Courant Institute, NYU | 数值分析、科学计算 | [个人主页](https://dyjdongyijun.github.io/) · [Google Scholar](https://scholar.google.com/citations?user=l3bmbCkAAAAJ&hl=en&oi=ao/) |

| **Diana Halikias** | Courant Institute, NYU | 数值线性代数、随机化方法 | [Simons Institute](http://simons.berkeley.edu/people/diana-halikias) |

| **Anna Ma** | UC Irvine, 数学系 | 随机化数值线性代数、优化、机器学习 | [个人主页](https://sites.google.com/view/annama/home) · [UCI Faculty](https://webapps.ps.uci.edu/faculty_profiles/m) |

| **Fabio Matti** | EPFL, 瑞士 | 数值方法、科学计算 | [个人主页](https://fmatti.github.io/) |

| **Deanna Needell** | UCLA, 数学系 | 压缩感知、随机化算法、机器学习、信号处理 | [UCLA主页](https://www.math.ucla.edu/~deanna) ·  [Wikipedia](https://en.wikipedia.org/wiki/Deanna_Needell) |

| **Katherine J. Pearce** | Oden Institute, UT Austin | 随机化数值线性代数 | [Oden主页](https://oden.utexas.edu/people/directory/Kate%20Pearce) · [个人主页](https://kjpearce.github.io/) |

| **Elizaveta Rebrova** | Princeton, ORFE | 随机矩阵理论、高维概率、数值线性代数 | [个人主页](https://erebrova.github.io/) · [Google Scholar](https://scholar.google.com/citations?user=nZ27XjIAAAAJ&hl=en) |

| **Disha Shur** | Purdue, 计算机系 | 计算机科学 | [Purdue](https://hammer.purdue.edu/authors/Disha_Shur/11892086) |

| **Rudi Smith** | Virginia Tech, 数学系 | 数值分析 | — |

| **Hai-Xiao Wang** | UW-Madison, 数学系 | 数学 | [个人主页](https://haixiaowang.github.io/) |

  

> **Deanna Needell** 是本文中最资深的作者之一，UCLA 数学教授、Dunn Family 数据理论讲席教授、IDRE 执行主任，发表 200+ 篇论文，也是 RNLA 2025 研讨会的组织者之一。

>

> **Katherine J. Pearce** 是 Oden Institute 的 Peter O'Donnell Jr 博士后，师从 Per-Gunnar Martinsson，获 NSF MPS-ASCEND 博士后奖学金。


## 报告大纲（约90分钟，覆盖论文 Section 1-5）

| 时间 | 内容 | 时长 |
|------|------|------|
| 前置知识 | Transformer 架构通俗讲解（Encoder/Decoder/Cross Attention） | ~15 min |
| Part 1 | 引言 + Attention 数学模型 (Section 1-2) | ~15 min |
| Part 2 | 稀疏/聚类方法 (Section 3) | ~20 min |
| Part 3 | 低秩近似方法 (Section 4) | ~20 min |
| Part 4 | 核方法 (Section 5) | ~15 min |
| Part 5 | 总结与讨论 | ~5 min |

> 论文 Section 6 (Latent Attention / MLA) 和 Section 7 (张量方法) 内容丰富，留作后续组会讨论。

---

## 前置知识：Transformer 架构通俗讲解

> 参考来源：[CSDN 博客](https://blog.csdn.net/m0_56997192/article/details/147339733)（基于李宏毅老师课程整理）

### 0.1 Transformer 是什么？

Transformer 本质上是一个 **Seq2Seq（序列到序列）模型**：输入一个序列，输出一个序列，输出的长度由模型自己决定。

典型应用：
- 机器翻译：输入中文句子 → 输出英文句子
- 语音识别：输入语音信号 → 输出文字
- 对话系统：输入用户问题 → 输出回答
- 甚至语法分析、物品识别等

整体结构就两大块：**Encoder（编码器）** 和 **Decoder（解码器）**。

**Transformer 整体架构图：**

```
                        输出概率
                           ↑
                      ┌─────────┐
                      │ Softmax │
                      └────┬────┘
                      ┌────┴────┐
                      │ Linear  │
                      └────┬────┘
                           │
                    ┌──────┴──────┐
                    │  Decoder    │
                    │  × N 层     │
                    │             │
                    │ ┌─────────┐ │
                    │ │ FFN +   │ │
                    │ │ Add&Norm│ │
                    │ ├─────────┤ │
                    │ │ Cross   │ │◄──── Encoder 输出（K, V）
                    │ │Attention│ │
                    │ │+Add&Norm│ │
                    │ ├─────────┤ │
                    │ │ Masked  │ │
                    │ │Self-Attn│ │
                    │ │+Add&Norm│ │
                    │ └─────────┘ │
                    └──────┬──────┘
                           │
                   输出嵌入 + 位置编码
                   (右移一位)


     ┌──────────────┐
     │   Encoder    │
     │   × N 层     │
     │              │
     │ ┌──────────┐ │
     │ │ FFN +    │ │
     │ │ Add&Norm │ │
     │ ├──────────┤ │
     │ │Self-Attn │ │
     │ │+Add&Norm │ │
     │ └──────────┘ │
     └──────┬───────┘
            │
    输入嵌入 + 位置编码
```

---

### 0.2 Encoder：把输入"读懂"

Encoder 的工作：接收一排向量（输入序列），输出一排向量（编码后的表示）。

Encoder 内部由多个 Block 堆叠而成，每个 Block 包含：

**Encoder 单个 Block 详细结构：**

```
          输入 x
          │
          ▼
   ┌──────────────┐
   │  Multi-Head   │
   │ Self-Attention │   每个 token 与所有 token 计算相关度
   └──────┬───────┘
          │
          ├──── + ────  x（残差连接：把原始输入加回来）
          │
          ▼
   ┌──────────────┐
   │    Layer      │
   │ Normalization │   对每个样本归一化，稳定训练
   └──────┬───────┘
          │ (记为 m)
          ▼
   ┌──────────────┐
   │  Feed Forward │
   │   Network     │   两层全连接：Linear → ReLU → Linear
   └──────┬───────┘
          │
          ├──── + ────  m（又一次残差连接）
          │
          ▼
   ┌──────────────┐
   │    Layer      │
   │ Normalization │
   └──────┬───────┘
          │
          ▼
       输出（送入下一个 Block）
```

几个关键点：
- **残差连接（Residual Connection）**：把输入直接加到输出上，防止信息丢失，也让梯度更好传播
- **Layer Normalization**：对每个样本做归一化，稳定训练
- **位置编码（Positional Encoding）**：在第一个 Block 之前加上，因为 Self-Attention 本身不知道 token 的先后顺序

> BERT 其实就是 Transformer 的 Encoder 部分。

---

### 0.3 Decoder：一个字一个字地"说出来"

Decoder 是**自回归（Autoregressive）**的，意思是它一个一个地生成输出：

```
步骤 1：输入 [BEGIN] 标记 → 输出第 1 个字（比如"机"）
步骤 2：输入 [BEGIN] + "机" → 输出第 2 个字（比如"器"）
步骤 3：输入 [BEGIN] + "机" + "器" → 输出第 3 个字（比如"学"）
...
步骤 n：当模型输出 [END] 标记时，停止生成
```

Decoder 的输出其实是一个概率向量，每个位置代表词表中某个字的概率，取概率最大的那个字作为输出。

Decoder 的结构和 Encoder 几乎一样，但有两个关键区别：

**区别 1：Masked Self-Attention（掩码自注意力）**

因为 Decoder 是一个一个生成的，在计算第 i 个位置的 attention 时，只能看到前 i 个位置，不能偷看后面的内容：
- 算 b¹ 时：只看 a¹
- 算 b² 时：只看 a¹、a²
- 算 b³ 时：只看 a¹、a²、a³
- ...

这就是论文中提到的**因果掩码（Causal Mask）**，对应 attention 矩阵变成下三角。

**Masked Attention 矩阵示意（4个token的例子）：**

```
         k₁   k₂   k₃   k₄
   q₁ [ 0.9  -∞   -∞   -∞  ]     q₁ 只能看 k₁
   q₂ [ 0.3  0.7  -∞   -∞  ]     q₂ 能看 k₁, k₂
   q₃ [ 0.1  0.2  0.7  -∞  ]     q₃ 能看 k₁, k₂, k₃
   q₄ [ 0.1  0.1  0.3  0.5 ]     q₄ 能看所有

   -∞ 经过 softmax 后变成 0，相当于完全屏蔽
```

**区别 2：最后多了 Linear + Softmax 层**

把 Decoder 的隐藏状态映射到词表大小的向量，再用 Softmax 得到每个字的概率。

---

### 0.4 Encoder 和 Decoder 之间怎么配合？—— Cross Attention

Encoder 和 Decoder 之间通过 **Cross Attention（交叉注意力）** 传递信息：

**Cross Attention 信息流：**

```
  Encoder 输出                          Decoder 当前层
  ┌─────────┐                          ┌─────────┐
  │ a₁ a₂ a₃│                          │  d_i    │
  └──┬──┬───┘                          └────┬────┘
     │  │                                   │
     ▼  ▼                                   ▼
   ┌──┐ ┌──┐                             ┌──┐
   │Wk│ │Wv│                             │Wq│
   └┬─┘ └┬─┘                             └┬─┘
    │    │                                 │
    ▼    ▼                                 ▼
   K,V 来自 Encoder                    Q 来自 Decoder
    │    │                                 │
    └────┼─────────────────────────────────┘
         │
         ▼
   ┌───────────────┐
   │  Attention:    │
   │  softmax(QK^T) │  →  加权求和 V  →  输出
   └───────────────┘
```

直觉理解：
- Decoder 问："我现在要生成什么？"（Query）
- Encoder 答："这是输入的全部信息，你自己挑重点"（Key + Value）

在原始 Transformer 中，Encoder 最后一层的输出会传给 Decoder 的每一层。

---

### 0.5 训练 vs 推理的区别

**训练时（Teacher Forcing）**：
- Decoder 每一步的输入用的是**真实标签**（ground truth），而不是上一步的预测结果
- 这样训练更稳定，但会导致训练和推理的 mismatch

**推理时**：
- 没有真实标签了，每一步的输入是上一步**真正的预测输出**
- 如果某一步预测错了，后面可能"一步错，步步错"

**解决方案 — Scheduled Sampling**：训练时偶尔故意给 Decoder 输入一些错误信息，让模型学会容错。

---

### 0.6 一些实用技巧

- **Copy Mechanism**：让模型可以直接从输入中复制内容到输出（适合人名、专有名词等）
- **Beam Search**：生成时不只贪心选概率最大的，而是保留 top-k 个候选路径，最后选整体最优的（但不一定总是更好）
- **Non-autoregressive (NAT) Decoder**：一次性并行输出所有 token，速度快但质量通常不如自回归

---

## Part 1: 引言与 Attention 数学模型（~20 min）

### 1.1 为什么关注 Attention 的效率？

**核心问题**: Attention 机制是 Transformer 的计算核心，但其复杂度关于序列长度 N 是 **O(N²)** 的，这是大规模推理的瓶颈。

**本文视角**: 从 **数值分析** 和 **数值线性代数** 的角度，系统性地审视和分类各种快速 Attention 近似方法。

**本文目标**:
1. 按数值原理对快速近似方法进行分类：稀疏/聚类、低秩/子空间投影、随机 sketching、张量分解
2. 在统一数学框架下呈现，桥接深度学习与计算数学两个领域

**与已有综述的区别**:
- [Tay+22] 和 [Ges+25] 提供 Transformer 的宏观视角
- 本文聚焦 Attention 机制本身，引入数值方法的分类体系
- 主要关注 **推理阶段**（inference），与训练优化解耦
- 不涉及实现层面优化（如 FlashAttention）

---

### 1.2 Attention 机制的数学定义

**输入**: N 个 token，每个维度 d，记为矩阵 X ∈ R^{N×d}

**三个线性映射**（通过预训练获得）:
- W^Q ∈ R^{d×d_head} （Query 权重）
- W^K ∈ R^{d×d_head} （Key 权重）
- W^V ∈ R^{d×d}     （Value 权重）

**计算流程**:
```
Q = X W^Q ∈ R^{N×d_head}    (Query 矩阵)
K = X W^K ∈ R^{N×d_head}    (Key 矩阵)
V = X W^V ∈ R^{N×d}         (Value 矩阵)
```

**Attention 分数矩阵**:
```
A = exp(Q K^T / √d_head) ∈ R^{N×N}
```
等价于核函数 κ_exp(q_i, k_j) = exp(⟨q_i, k_j⟩ / √d_head)

**归一化**: Z 为对角矩阵，Z(i,i) = Σ_j A(i,j)

**输出**: Y = Z^{-1} A V ∈ R^{N×d}

**Self-Attention 单个 token 的计算流程图：**

```
  输入 token x_i
       │
  ┌────┼────────────────┐
  ▼    ▼                ▼
 W^Q  W^K              W^V
  │    │                │
  ▼    ▼                ▼
 q_i  k_i              v_i        ← 每个 token 生成 3 个向量
  │    │                │
  │    │                │
  ▼    ▼                │
  q_i · k_1^T           │         ← q_i 和所有 k 做点积
  q_i · k_2^T           │
  q_i · k_3^T           │
  ...                   │
  q_i · k_N^T           │
       │                │
       ▼                │
  ÷ √d_head            │         ← 缩放，防止点积太大
       │                │
       ▼                │
   softmax              │         ← 变成概率分布（和为1）
   [α₁, α₂, ..., αN]   │
       │                │
       ▼                ▼
   α₁·v₁ + α₂·v₂ + ... + αN·vN  ← 对所有 value 加权求和
       │
       ▼
     y_i（输出）
```

> **讲解要点**: 每个 token i 用 query 与所有 key 比较，得到概率分布，然后对 value 加权平均。复杂度 O(N²(d + d_head))，关于 N 是二次的。

**因果掩码**: next-token prediction 中加掩码 M_{ij} = δ_{j≤i}，A 变为下三角。

---

### 1.3 Multi-Headed Attention (MHA) 及其变体

**MHA**: N_heads 个并行 attention head，每个有独立的 W^Q_h, W^K_h, W^V_h

**Multi-Head Attention 结构图：**

```
                    输入 X
         ┌──────────┼──────────┐
         ▼          ▼          ▼
      ┌──────┐  ┌──────┐  ┌──────┐
      │Head 1│  │Head 2│  │ ... │  │Head H│
      │Q₁K₁V₁│  │Q₂K₂V₂│       │QₕKₕVₕ│
      └──┬───┘  └──┬───┘       └──┬───┘
         │         │              │
         ▼         ▼              ▼
       Y₁         Y₂    ...     Yₕ     ← 每个 head 独立算 attention
         │         │              │
         └────┬────┴──────────────┘
              │
              ▼
        Concat(Y₁, Y₂, ..., Yₕ)        ← 拼接所有 head 的输出
              │
              ▼
           × W^O                         ← 线性投影回原始维度
              │
              ▼
            输出 O
```

每个 head 关注不同的"语义子空间"，比如 head 1 关注语法关系，head 2 关注语义相似度。
```
对每个 head h: Q_h, K_h, V_h → A_h → Y_h = Z_h^{-1} A_h V_h
最终输出: O = (Y_1 | ... | Y_{N_heads}) W^O
```
通常设 N_heads × d_head = d

**Grouped Query Attention (GQA)**: 多个 query head 共享一组 KV head
- 减少 KV cache 内存，已成为现代模型标配（Llama 3, Gemma 3, Qwen3, DeepSeek-R1）

**Multi-Query Attention (MQA)**: GQA 极端情况，N_groups = 1

| 模型 | KV Cache 大小 | 计算复杂度 |
|------|--------------|-----------|
| MHA | 2N·N_heads·d_head | O(N²d) |
| GQA | 2N·N_groups·d_head | O(N²d) |
| MQA | 2N·d_head | O(N²d) |

> **讲解要点**: 计算复杂度都是 O(N²d)，GQA 优势在于减少内存和通信开销。文本数据被认为是稀疏和低秩的，这激发了后续所有加速方法。

---

## Part 2: 稀疏与聚类方法（~25 min）

### 2.1 核心思想

**观察**: softmax 操作会放大矩阵元素间的差异 → 实际中 attention 矩阵 Z^{-1}A 近似稀疏（论文 Figure 5 展示了 Llama 3.2 的稀疏模式）

**关键思路**: 只计算最"重要"的 QK^T 条目（heavy hitters），廉价近似其余条目
- 可将 O(N²) 降至 O(N log N) 或 O(N)

**数学表达**: 对第 i 个 query q_i:
```
o_i = Σ_{j∈P_i} exp(⟨q_i, k_j⟩/√d_head - z(i, P_i)) · v_j
```
其中 P_i ⊆ [N] 是 q_i "关注"的 key 的子集（标准 attention 中 P_i = [N]）

**三个核心问题**:
1. 如何高效找到重要的 key-query 对的聚类？
2. 如何保证聚类大小均衡？
3. 如何近似归一化矩阵 Z 和剩余条目？

---

### 2.2 Locality Sensitive Hashing (LSH)

经典聚类（如 k-means）太贵，因此很多方法使用 **局部敏感哈希** 做近似聚类。

**定义**: 哈希函数 h: R^{d_head} → B，满足"近邻向量大概率映射到同一桶"

**三种 LSH 构造**:

**(a) 角度 LSH** (Reformer 使用):
```
h(x) = argmin_i [H(x)]_i
H(x) = [⟨w_1,x⟩, ..., ⟨w_k,x⟩, -⟨w_1,x⟩, ..., -⟨w_k,x⟩]
```
映射到 2^k 个桶

**(b) 超平面 LSH** (HyperAttention, KDEFormer 使用):
```
h(x) = (1{⟨w_1,x⟩>0}, ..., 1{⟨w_k,x⟩>0})
```

**(c) 随机投影 LSH** (SMYRF 使用):
```
h(x) = ⌊(⟨w,x⟩ + b) / r⌋
```

> **讲解要点**: 多轮 LSH 并行运行再合并，可提高聚类质量。

---

### 2.3 基于聚类的方法

#### Reformer [KKL20]
- **特点**: 共享 QK（W^Q = W^K，即 Q = K）
- **方法**: 用角度 LSH 将 key/query 分桶 → 桶内重排 → 分成大小为 m=2N/b 的 chunk → 每个 query 只关注自己 chunk 和前一个 chunk 的 key
- **多轮**: 2-4 轮 LSH 即可获得较高精度，8 轮接近完美
- **复杂度**: 从 O(N²) 降低（取决于桶大小）

#### Routing Transformer [Roy+21]
- **特点**: Q ≠ K 的情况
- **方法**: 用 mini-batch k-means 对 Q 和 K 聚类，学习质心 μ_1,...,μ_k
- 每个质心选 top-k 最近的 Q 行和 K 行 → 只计算同一聚类内的 attention
- **质心数**: 约 √N，平衡聚类开销和点积计算 → 复杂度 O(N^{1.5} d_head)

#### SMYRF [Dar+20]
- **特点**: 平衡聚类 + 多轮合并
- **方法**: 构造变换 φ, ψ 使得 ‖φ(·)-ψ(k)‖ 保持与 ⟨·,k⟩ 相同的排序
- 多轮 LSH 后，按 softmax 质量加权合并各轮结果

#### Multipole Attention [Hoo+25]
- **特点**: 层次化近似，类似多极展开
- **方法**: k-means 聚类 key → 高分聚类精确计算 → 低分聚类用质心近似
- 近似: N_j · exp(⟨q, k^c_j⟩) · v^c_j （用聚类质心代替）
- 支持层次化递归，在线聚类（滑动窗口 k-means）

---

### 2.4 LSH + 采样的组合方法

#### KDEFormer [Zan+23]
- **核心创新**: 将 dot-product attention 关联到高斯核密度估计 (KDE)
- **近似**: Y ≈ Z̃^{-1} A S^T S V
  - Z̃: 通过快速高斯 KDE 近似归一化矩阵
  - S: 基于 KDE 近似构造的采样矩阵（m = N^{1-Ω(1)} ≪ N）
- **理论保证**: 复杂度 O(ε^{-2} d · N^{1.173+o(1)})，误差 ‖Y - Ỹ‖_op ≤ ε · ‖Z^{-1}A‖_op · ‖V‖_op
- **LSH 加速**: 用 LSH 找到 A 的"重"元素 A_spar，对残差 A_res = A - A_spar 做采样（降低 stable rank）

#### HyperAttention [Han+23]
- **方法**: LSH (Hamming 排序) + leverage score 采样
- **步骤**:
  1. LSH 对 Q, K 行排序 → 定义大小为 b 的块 → 同块内精确计算
  2. 随机选 ℓ 个 key 索引，也精确计算
  3. 其余设为 0
  4. 采样矩阵 S 基于 V 的行范数平方
- **近似**: Z̃^{-1} A S^T · S V
- **改进**: Prescoring [Li+25] 用 k-means 或 leverage score 预选 key

> **讲解要点**: KDEFormer 和 HyperAttention 的共同模式是 "找重要条目 + 采样近似矩阵乘法"，区别在于如何构造 Z̃ 和 S。

| 模型 | 重要性采样 | 近似方法 |
|------|-----------|---------|
| Reformer | 多轮角度 LSH, Q=K | 桶内精确计算 |
| Routing | k-means 聚类 | 聚类内精确计算 |
| SMYRF | 多轮 LSH 平衡聚类 | 加权合并各轮 |
| Multipole | k-means + 质心 | 精确+质心近似 |
| KDEFormer | KDE + LSH | Z̃^{-1}AS^TSV |
| HyperAttention | Leverage score + LSH | Z̃^{-1}AS^TSV |

---

## Part 3: 低秩近似方法（~20 min）

### 3.1 低秩结构的实验观察

**关键观察**（论文 Figure 7, 8，基于 Llama 3.2 实验）:
- Q_h, K_h 本身就是低秩的（d_head ≪ N），且奇异值有中等衰减
- A_h = exp(Q_h K_h^T / √d_head) 的奇异值衰减很快
- Z^{-1}_h A_h 也有较弱但明显的衰减

→ 可以用低秩分解来近似 attention 矩阵

---

### 3.2 压缩预训练模型的方法

#### Loki [Sin+24] — 低秩 Key 压缩
- **观察**: K 矩阵的"有效秩"（解释 90% 方差的主成分数）约为 80，而 d=128
- **方法**:
  1. 离线计算 K 的主成分矩阵 P_r ∈ R^{d_head×r}（r = d_head/4 或 d_head/8）
  2. 用 P_r 近似 attention score: softmax(q_i^T P_r (K P_r)^T / √d_head) V
  3. 选 top-k key（k = N/8 或 N/4）
  4. 对 top-k key 重新精确计算
- **复杂度**: O(N²r) 代替 O(N²d_head)，仍是二次但常数更小
- **优点**: 不需要重训练或微调

#### Skyformer [Che+21b] & WILDCAT [SM26] — 低秩分解 A
- **思路**: A = κ_exp(Q, K)，构造对称 PSD 核矩阵 B ∈ R^{2N×2N}:
```
B = [κ_exp(Q,Q)    A      ]
    [A^T         κ_exp(K,K)]
```
- 对 B 做低秩近似 B̂ → 提取 Â 作为 A 的近似
- **Skyformer**: 用 Nyström 近似；使用高斯核 κ_Gauss 避免数值溢出
- **WILDCAT**: 用随机 pivoted Cholesky 分解；利用 Q→τQ, K→τ^{-1}K 的不变性优化低秩性

> **讲解要点**: Loki 是"先粗筛再精算"的两阶段策略；Skyformer/WILDCAT 是直接对核矩阵做低秩分解。

---

### 3.3 替代 Attention 机制

#### Linformer [Wan+20]
- **最早的线性复杂度方法之一**
- 引入投影矩阵 P_K, P_V ∈ R^{N×k}（k ≪ N，训练时学习）:
```
softmax(Q (P_K K)^T / √d_head) (P_V V)
```
- k 越小，内存和时间复杂度越低
- 理论假设较强，实验效果尚可

#### Nyströmformer [Xio+21]
- 用 landmark query/key 做 Nyström 类近似:
```
softmax(Q(P_K K)^T/√d) · [softmax((P_Q Q)(P_K K)^T/√d)]^† · softmax((P_Q Q)K^T/√d) · V
```
- P_Q, P_K 通过 segment-means（相邻行取平均）计算
- 经验上 64 个 segment 通常足够

> **讲解要点**: Linformer 直接学投影矩阵，Nyströmformer 借鉴了数值线性代数中的 Nyström 方法来近似核矩阵。

---

## Part 4: 核方法（~20 min）

### 4.1 核方法视角下的 Attention

**统一公式**: attention 输出可以写成核加权平均:
```
y_i = Σ_j κ(q_i, k_j) v_j^T / Σ_{j'} κ(q_i, k_{j'})
```
标准 attention 使用指数核 κ(q,k) = exp(⟨q,k⟩/√d_head)

**线性化的关键**: 如果存在有限维特征映射 φ: R^{d_head} → R^M 使得 κ(q,k) = ⟨φ(q), φ(k)⟩，则:
```
y_i = φ(q_i)^T [Σ_j φ(k_j) v_j^T] / [φ(q_i)^T Σ_{j'} φ(k_{j'})]
```
先算 Σ_j φ(k_j)v_j^T → 复杂度 O(NM²)，关于 N 线性！

**问题**: 指数核没有有限维特征映射 → 需要近似

---

### 4.2 多项式核方法

**多项式核**: κ(q,k) = ⟨q,k⟩^p

#### PolySketchFormer [KMZ23]
- 用偶数次多项式核替代 softmax:
```
A^(p)(i,j) = ⟨q_i, k_j⟩^p / (1 + Σ_{j'} ⟨q_i, k_{j'}⟩^p)
```
- **p 的效果**: p=0 → 均匀分布；p→∞ → hardmax（只关注最大内积的 token）
- **线性化**: 利用 Kronecker 积 ⟨q,k⟩^p = ⟨q^{⊗p}, k^{⊗p}⟩
  - 直接计算: O(N d_head^{p+1})，对 N 线性但 d_head 指数增长
  - **Sketching 解决**: 用近似矩阵乘法 (AMM)，sketching 维度 r ∈ {32, 64}
  - 递归算法 + 自张量化保证非负性

#### LevAttention [Kan+24]
- 用 **leverage score**（杠杆分数）找 heavy hitter
- 定义 f-sensitivity: α_j^f = sup_y f(⟨k_j, y⟩) / Σ_ℓ f(⟨k_ℓ, y⟩)
- 集合 U = {i: α_i^f > ε}，|U| ≤ d^{p/2}/ε（不依赖 N！）
- p=2 时可通过 K 的 QR 分解计算

#### Tensor Sketch [PP25]
- 用 CountSketch 近似高维张量积，避免显式构造 q^{⊗p}
- 核心: FFT^{-1}(FFT(C_1 x) ⊙ ... ⊙ FFT(C_p x))
- 复杂度 O(d_head + r log r)，方差 ≤ (3^p - 1)/r · ‖q‖^{2p}‖k‖^{2p}

---

### 4.3 Performer: 随机核特征

**核心思想**: 构造随机特征映射 φ: R^{d_head} → R^r_+，使得 E[⟨φ(q), φ(k)⟩] = κ_exp(q,k)

**Random Orthogonal Positive Features (ROPF)**:
```
φ(x) = h(x)/√N_R · [f_1(ω_1^T x), ..., f_1(ω_{N_R}^T x), ..., f_{N_f}(ω_1^T x), ..., f_{N_f}(ω_{N_R}^T x)]
```
r = N_R · N_f

**三种随机特征**:
- φ_trig: h(x) = exp(‖x‖²/2), f = {sin, cos} — 基于 Random Fourier Features
- φ_+: h(x) = exp(-‖x‖²/2), f = {exp} — 保证正性
- φ_hyp+: h(x) = exp(-‖x‖²/2)/√2, f = {exp(·), exp(-·)} — 基于双曲余弦

**计算**: 先算 K'^T V，再算 Q'(K'^T V) → O(Nrd) 线性复杂度

**KV cache**: 可压缩到 O(N_R N_f (d+1))

**局限**: 随机投影会模糊尖锐分布 → Scatterbrain 结合 φ_+ 和 LSH 来同时利用低秩性和稀疏性

> **讲解要点**: Performer 是唯一真正实现 O(N) 复杂度的方法，但代价是对尖锐 attention 分布的近似质量较差。

---

## Part 5: 总结与讨论（~5 min）

### 前五节方法全景图

**Attention 的 O(N²) 瓶颈在哪？**

```
标准 Attention 计算流程：

  Q (N×d)    K (N×d)         V (N×d)
    │          │                │
    └────┬─────┘                │
         ▼                      │
    Q × K^T                     │
   (N × N)  ← ★瓶颈！N²量级★   │
         │                      │
         ▼                      │
     softmax                    │
    (N × N)                     │
         │                      │
         └──────────┬───────────┘
                    ▼
              A × V = Y
              (N × d)
```

**各类方法的加速思路：**

```
                    ★ N×N 矩阵太大了！★
                          │
            ┌─────────────┼─────────────┐
            │                           │
     直接近似这个矩阵              换一种算法绕过它
            │                           │
     ┌──────┼──────┐             ┌──────┼──────┐
     │      │      │             │      │      │
   稀疏   低秩   采样          投影    核方法
   只算   分解成  随机抽样     把N维   用特征映射
   重要   小矩阵  几行来算    压到k维  改变乘法顺序
   的条目  的乘积              (k≪N)  先算K^T·V
     │      │      │             │      │
  Reformer Loki  KDEFormer  Linformer Performer
  Routing  Sky-  Hyper-     Nyström-  PolySketc-
  SMYRF   former Attention  former   hFormer
  Multipole WILDCAT                  LevAttention
```

### 各类方法对比

| 方法类别 | 复杂度 | 是否需要重训练 | 核心数学工具 |
|---------|--------|--------------|-------------|
| 聚类/LSH | O(N√N) ~ O(N log N) | 部分需要 | LSH, k-means |
| LSH+采样 | O(N^{1.17}) | 否 | KDE, AMM, leverage score |
| 低秩 | O(N²r) 或 O(NM²) | 部分需要 | SVD, Nyström, Cholesky |
| 核方法 | O(Nr²) ~ O(Nrd) | 部分需要 | 随机特征, sketching, FFT |

### 未涉及的内容（论文后半部分）

- **Section 6 — Latent Attention (MLA)**: DeepSeek 提出的共享潜在空间 KV 压缩机制，以及 TransMLA 模型转换方法
- **Section 7 — 张量方法**: 权重张量化压缩（CP/Tucker 分解）、张量化 Attention 模型（捕获高阶交互）、张量输入的 Attention

### 讨论环节建议问题

1. 这些方法中，哪些最有可能在实际大模型中部署？（目前 GQA 已经在用）
2. 稀疏方法和低秩方法能否有效结合？（Scatterbrain 做了初步尝试）
3. 从数值分析的角度，Attention 矩阵的低秩性和稀疏性有没有更深层的理论解释？
4. 随着上下文窗口越来越长（100K+），哪类方法最有前景？
