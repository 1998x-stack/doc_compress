## PositionRank 文档压缩思路

**PositionRank** 是一种基于词语在文档中的位置来增强关键词权重的算法，它与传统的基于词频（TF）或上下文关系（如 TF-IDF 和 TextRank）的算法不同，PositionRank 更注重词语在文档中的相对位置。该算法的核心思想是通过分析词语在文档中的位置（如词语在句子或段落中的顺序）来增强关键词的重要性。

### 1. 简介

PositionRank 结合了词语的 **位置权重** 和 **PageRank** 排序机制，从而能够有效地压缩文档内容，并突出显示具有高信息量的关键部分。与传统算法相比，PositionRank 更加注重词汇出现的顺序和上下文结构，能够更好地理解文档的层次和语义关联。

### 2. 带 QUERY 的 PositionRank 文档压缩

当文档中有查询（**QUERY**）时，PositionRank 旨在提取与查询最相关的部分，通过计算词语在文档中的 **位置权重** 和 **PageRank 值**，结合查询关键词的相关性，选取最重要的文档分块。

#### 步骤

1. **对 QUERY 进行分词**  
   使用分词工具（如 Jieba）对查询内容进行分词，得到查询中的关键词列表。

2. **对文档进行分块**  
   将文档分为多个块（例如按段落、句子或固定字数分块），并为每个分块分配唯一的序号。

3. **对每个分块进行分词**  
   对每个文档块单独进行分词，得到块内的词汇列表。

4. **计算每个词的本地位置得分（PositionRank 核心）**  
   - **词频统计**：统计每个分块中词汇的出现频率（即 **TF**）。
   - **位置权重计算**：每个词在分块中的位置对其重要性有影响。一般来说，词汇在分块中的位置越靠前，其重要性越高。PositionRank 通过将每个词语的位置的倒数求和作为该词的 **位置权重**。
     $$
     \text{位置权重} = \sum_{i=1}^{n} \frac{1}{\text{位置}_i}
     $$
     其中，位置 $ \text{位置}_i $ 是词语在分块中的出现位置。

5. **构建关键词图并计算得分**  
   - **节点**：每个分块中的词汇作为图中的节点。
   - **边权重**：边的权重可以基于词汇的 **词频** 和 **位置权重** 进行加权。具体来说，如果词 $ w_i $ 和 $ w_j $ 出现于同一分块，它们之间的边权重可以通过以下方式计算：
     $$
     \text{边权重} = \text{TF}_i \times \text{TF}_j \times \left( \frac{1}{\text{位置}_i} + \frac{1}{\text{位置}_j} \right)
     $$
     这种加权方式可以确保位置较前的词对图中的其他词的影响更大。

6. **使用 PageRank 计算节点权重**  
   通过 **PageRank** 算法对构建的关键词图进行计算，得到每个词汇的 **PageRank** 权重。PageRank 会根据词汇在分块中的重要性和词汇之间的连接关系来调整词汇的权重，进一步突出关键词汇。

7. **全局关键词调整**  
   - 根据文档的整体结构，汇总每个分块中词语的 **PageRank 值**，并结合词的全局分布调整其权重。通过这种方式，能够更好地反映词语在文档中的重要性。
   - 综合每个分块内词语的权重来评估分块的整体重要性。

8. **选择与 QUERY 相关的分块**  
   计算每个分块与查询关键词的相关性（例如，通过查询关键词的 **TF-IDF** 或 **余弦相似度**），选择与查询最相关的 **TOP-N** 个分块。

9. **压缩文档并按序号排序输出**  
   根据计算的分块重要性和查询相关性输出压缩后的文档，按原始序号重新排序，保留最重要的分块。

#### 优势

- **查询驱动**：PositionRank 能够根据查询关键词的相关性来引导文档压缩过程，确保输出的文档更加符合查询需求。
- **位置权重增强**：通过引入词语的 **位置权重**，能够更好地理解文档中的信息层次，提高词汇的重要性评估。
- **全局优化**：结合 **PageRank** 算法，PositionRank 能够综合考虑词汇在文档中的位置和权重，提取最具信息量的部分。

### 3. 不带 QUERY 的 PositionRank 文档压缩

在没有查询（**QUERY**）的情况下，PositionRank 将自动基于文档内容，通过分析词语的 **位置权重** 和 **PageRank**，提取出文档中最具信息量和重要性的部分。

#### 步骤

1. **对文档进行分块**  
   将文档分为多个块（例如按段落、句子或固定字数分块）。

2. **对每个分块进行分词**  
   对每个文档块单独进行分词，得到块内的词汇列表。

3. **计算每个词的本地位置得分（PositionRank 核心）**  
   - **词频统计**：统计每个分块中词汇的出现频率（即 **TF**）。
   - **位置权重计算**：每个词在分块中的位置对其重要性有影响。通过计算词语位置的倒数求和得到每个词的 **位置权重**。

4. **构建关键词图并计算得分**  
   - **节点**：每个分块中的词汇作为图中的节点。
   - **边权重**：边的权重根据 **TF** 和 **位置权重** 计算。
   
5. **使用 PageRank 计算节点权重**  
   通过 **PageRank** 算法对关键词图进行计算，得到每个词汇的 **PageRank** 权重。

6. **全局关键词调整**  
   汇总每个分块中词语的 **PageRank** 值，结合词语的全局分布，调整词语在图中的权重。

7. **选择 TOP-N 分块**  
   根据计算的分块重要性，选择 **TOP-N** 个最重要的分块，这些分块包含文档的核心信息。

8. **压缩文档并排序输出**  
   输出所选的 **TOP-N** 个分块，按原始序号重新排序，生成压缩后的文档。

#### 优势

- **自动提取核心信息**：在没有查询的情况下，PositionRank 能够自动提取出文档最具信息量的部分，适用于自动摘要和信息提取任务。
- **位置权重增强**：通过 **位置权重**，PositionRank 更加注重词语在文档中的位置，能够更好地理解文档的结构和信息层次。
- **全局优化**：通过 **PageRank** 算法，能够对文档中的关键词进行综合评估，从而提取最有信息量的内容。

### 4. 对比

| 方案                      | 带 **QUERY** 的压缩                          | 不带 **QUERY** 的压缩                   |
|-------------------------|-----------------------------------------|---------------------------------------|
| **输入**                | 包含查询（QUERY）                        | 仅包含文档文本                        |
| **目标**                | 提取与查询相关的文档部分                    | 提取文档中的关键信息块                  |
| **关键步骤**            | 计算词语的 **位置权重** 和 **PageRank**，结合查询相关性选择分块 | 计算词语的 **位置权重** 和 **PageRank**，选择最具信息量的分块 |
| **输出**                | 按照查询选择 TOP N 分块，排序输出              | 按照全局关键词选择 TOP N 分块，排序输出 |
| **优点**                 | 精准匹配查询需求，压缩文档保留查询相关部分         | 自动提取文档最有信息量的部分，无需外部输入      |

### 5. 总结

- **带 QUERY 的 PositionRank 文档压缩** 通过结合查询关键词，引导文档压缩过程，确保压缩后的文档更加贴合查询需求。
- **不带 QUERY 的 PositionRank 文档压缩** 通过分析文档内容和词语的 **位置权重**，自动提取出文档最具信息量的部分，适用于自动摘要和信息提取任务。