# 大模型文件结构概览

本节用于梳理一个大语言模型在本地或模型仓库中的常见文件结构。这里先以 GPT-2 为例进行说明，因为 GPT-2 的文件组成相对经典，适合理解模型配置、模型权重、分词器文件和推理配置之间的关系。

现代大模型在整体思路上仍然类似：通常都包含模型结构配置、模型权重、分词器配置、生成配置和说明文档。不过，相比 GPT-2，新的开源大模型更常见以下变化：

- 权重文件更多使用 `safetensors` 格式，而不是早期常见的 `pytorch_model.bin`
- 模型权重通常会被拆分成多个分片文件，并配套索引文件
- 分词器通常会提供统一的 `tokenizer.json`
- 部分模型会额外提供聊天模板、特殊 token 配置、量化权重或推理框架专用文件
- 模型配置中可能出现 RoPE、GQA/MQA、MoE、KV cache、滑动窗口注意力等新架构相关参数

## GPT-2 典型文件结构

```text
gpt2/
├── config.json
├── generation_config.json
├── model.safetensors 或 pytorch_model.bin
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
├── merges.txt
├── special_tokens_map.json
└── README.md
```

说明：不同来源的 GPT-2 模型文件可能不完全一致。例如早期版本可能主要使用 `pytorch_model.bin`，而较新的模型仓库可能同时提供或优先提供 `model.safetensors`。

## `config.json`：模型结构配置文件

### 文件作用

用于描述模型的网络结构和关键超参数。加载模型时，框架会先读取该文件，知道应该构建什么样的模型结构，然后再加载权重文件。

### GPT-2 中常见参数

- `model_type`：模型类型，例如 `gpt2`
- `architectures`：模型类名称，例如 `GPT2LMHeadModel`
- `n_embd`：隐藏层维度，也就是 token embedding 和 Transformer 隐状态的维度
- `n_layer`：Transformer block 层数
- `n_head`：注意力头数量
- `n_positions`：模型支持的位置编码长度
- `n_ctx`：上下文窗口长度，GPT-2 中通常和 `n_positions` 接近或一致
- `vocab_size`：词表大小
- `activation_function`：前馈网络中的激活函数
- `resid_pdrop`：残差连接 dropout 概率
- `embd_pdrop`：embedding dropout 概率
- `attn_pdrop`：attention dropout 概率
- `layer_norm_epsilon`：LayerNorm 中用于数值稳定的小常数
- `initializer_range`：参数初始化范围
- `bos_token_id`：开始 token 的 ID
- `eos_token_id`：结束 token 的 ID

### 文件内容

```json
{
  "activation_function": "gelu_new",    // 激活函数【激活函数负责引入“非线性”思考（类似于：如果信号低于某个阈值，就忽略它；如果高于某个阈值，就兴奋地传导下去）】
  "architectures": [    // 模型类名称
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,    // attention dropout 概率
  "bos_token_id": 50256,    // 开始 token 的 ID
  "embd_pdrop": 0.1,    // embedding dropout 概率
  "eos_token_id": 50256,    // 结束 token 的 ID
  "initializer_range": 0.02,    // 参数初始化范围
  "layer_norm_epsilon": 1e-05,  // LayerNorm 中用于数值稳定的小常数
  "model_type": "gpt2", // 模型类型
  "n_ctx": 1024,    // 上下文窗口长度，GPT-2 中通常和 `n_positions` 接近或一致
  "n_embd": 768,    // 隐藏层维度，也就是 token embedding 和 Transformer 隐状态的维度
  "n_head": 12, // 注意力头数量
  "n_layer": 12,    // 层数
  "n_positions": 1024,  // 模型支持的位置编码长度
  "resid_pdrop": 0.1,   // 残差连接 dropout 概率
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 50
    }
  },
  "vocab_size": 50257   // 词表大小【模型只认识50257个token】
}
```

### 需要重点理解的问题

- 模型结构参数和权重文件必须匹配
- 修改 `config.json` 不等于修改模型能力
- 上下文长度、词表大小、隐藏层维度等参数会直接影响模型能否正确加载权重

## `model.safetensors` / `pytorch_model.bin`：模型权重文件

### 文件作用

保存模型训练后得到的参数，例如 embedding、attention、MLP、LayerNorm 和输出层权重。

### 常见格式

- `pytorch_model.bin`：PyTorch pickle 序列化格式，早期模型仓库中非常常见
- `model.safetensors`：更安全、更适合快速加载的张量存储格式，现代模型更常见

### 需要重点理解的问题

- 权重文件通常是模型仓库中体积最大的文件
- 权重文件只保存参数，不单独描述完整模型结构
- 加载权重前需要先根据 `config.json` 构建模型结构
- 大模型权重可能拆分为多个文件，例如 `model-00001-of-000xx.safetensors`
- 分片权重通常会配套 `model.safetensors.index.json`

## `vocab.json`：词表文件

### 文件作用

保存 token 到 token ID 的映射关系。GPT-2 使用 BPE 分词方式，`vocab.json` 是理解文本如何被转换成数字 ID 的关键文件之一。

### 常见内容

- token 字符串
- token 对应的整数 ID
- 特殊 token 的 ID 映射

### 需要重点理解的问题

- 模型输入不是直接处理原始文本，而是处理 token ID
- `vocab_size` 应与模型配置和 embedding 权重匹配
- 词表变化会影响模型输入输出含义，不能随意替换

## `merges.txt`：BPE 合并规则文件

### 文件作用

保存 Byte Pair Encoding 的合并规则。GPT-2 的 tokenizer 会结合 `vocab.json` 和 `merges.txt`，将文本切分成 token。

### 常见内容

- 文件开头通常包含版本或注释信息
- 后续每一行表示一个可合并的 token pair
- 合并规则顺序代表优先级

### 需要重点理解的问题

- `vocab.json` 负责 token 到 ID 的映射
- `merges.txt` 负责文本切分时的合并规则
- 两者必须配套使用

## `tokenizer.json`：统一分词器文件

### 文件作用

保存完整的 tokenizer 配置，通常包含词表、合并规则、预处理、后处理和特殊 token 等信息。现代模型更倾向提供该文件，方便不同语言或框架统一加载 tokenizer。

### 常见内容

- tokenizer 模型类型
- normalizer 配置
- pre-tokenizer 配置
- post-processor 配置
- vocab 信息
- merges 信息
- special tokens 信息

### 需要重点理解的问题

- `tokenizer.json` 可以看作更完整、更统一的 tokenizer 描述
- 有些模型仍会同时保留 `vocab.json` 和 `merges.txt`
- 加载 tokenizer 时应优先使用模型仓库配套文件

## `tokenizer_config.json`：分词器加载配置

### 文件作用

用于描述 tokenizer 的加载方式和默认行为。它不一定包含完整词表，但会告诉框架如何初始化 tokenizer。

### 常见参数

- `tokenizer_class`：分词器类名
- `model_max_length`：tokenizer 侧的最大长度配置
- `bos_token`：开始 token
- `eos_token`：结束 token
- `unk_token`：未知 token
- `pad_token`：填充 token
- `add_prefix_space`：是否在分词前补充前导空格，GPT-2 相关任务中较常见

### 需要重点理解的问题

- tokenizer 的最大长度和模型真实上下文长度需要区分
- 特殊 token 配置会影响训练、推理和 padding 行为
- 聊天模型中还可能包含 `chat_template`

## `special_tokens_map.json`：特殊 Token 映射文件

### 文件作用

保存特殊 token 的名称和实际 token 字符串之间的映射。

### 常见内容

- `bos_token`
- `eos_token`
- `unk_token`
- `sep_token`
- `pad_token`
- `cls_token`
- `mask_token`

### 需要重点理解的问题

- 不同模型对特殊 token 的定义可能不同
- GPT-2 早期设计中没有独立的 padding token
- 如果为 GPT-2 人工添加 `pad_token`，需要注意 embedding 和训练/推理逻辑是否匹配

## `generation_config.json`：文本生成配置文件

### 文件作用

保存模型生成文本时的默认参数。它影响的是推理阶段的生成策略，而不是模型结构本身。

### 常见参数

- `max_length`：生成文本的最大总长度
- `max_new_tokens`：最多生成的新 token 数量
- `do_sample`：是否启用采样
- `temperature`：采样温度，控制随机性
- `top_k`：只从概率最高的前 K 个 token 中采样
- `top_p`：核采样阈值
- `repetition_penalty`：重复惩罚
- `num_beams`：beam search 的 beam 数量
- `bos_token_id`：开始 token ID
- `eos_token_id`：结束 token ID
- `pad_token_id`：填充 token ID

### 需要重点理解的问题

- 生成配置可以在推理代码中覆盖
- 生成参数会显著影响输出风格、稳定性和多样性
- `temperature`、`top_p`、`top_k` 通常需要结合具体任务调试

## `README.md`：模型卡与使用说明

### 文件作用

模型仓库中的 README 通常也被称为 Model Card，用于说明模型来源、用途、训练数据、限制、评测结果和使用方式。

### 常见内容

- 模型简介
- 适用场景
- 使用示例
- 训练数据说明
- 模型限制和风险
- 评测指标
- 引用方式
- License 信息

### 需要重点理解的问题

- README 是判断模型是否适合使用的重要入口
- 需要关注模型 License、商用限制和安全风险
- 对于学习项目，可以在 README 中记录模型实验结论

## 现代大模型常见新增文件

### 分片权重和索引文件

大模型参数量较大时，权重文件通常会拆分为多个分片：

```text
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
model.safetensors.index.json
```

索引文件用于记录每个参数张量存放在哪个分片中。

### 聊天模板

聊天模型通常需要将多轮对话转换成模型能理解的 prompt 格式。该逻辑可能保存在：

- `tokenizer_config.json`
- `chat_template.jinja`
- 推理框架的专用配置文件

### 量化权重文件

为了降低显存和推理成本，现代模型可能提供量化版本，例如：

- INT8
- INT4
- GPTQ
- AWQ
- GGUF

量化文件通常和原始全精度权重不能混用，需要使用对应的加载方式或推理框架。

### 推理框架专用文件

部分模型仓库可能包含服务部署或推理框架相关文件，例如：

- `config.pbtxt`
- `params.json`
- `adapter_config.json`
- `adapter_model.safetensors`
- `*.gguf`

这些文件通常服务于特定框架、LoRA 适配器或本地推理工具。

## 后续学习重点

- 理解 tokenizer 如何把文本转换为 token ID
- 理解 `config.json` 和模型结构之间的关系
- 理解权重文件和模型参数的对应关系
- 学会区分训练参数、模型结构参数和推理生成参数
- 对比 GPT-2 与现代开源大模型的文件组织差异
- 尝试手动加载 GPT-2，并查看每个文件在加载流程中的作用

## 参考资料

- Hugging Face GPT-2 模型仓库：https://huggingface.co/gpt2
- Hugging Face Transformers 文档：https://huggingface.co/docs/transformers
- Safetensors 文档：https://huggingface.co/docs/safetensors
