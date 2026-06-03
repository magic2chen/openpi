# OpenPI π₀ PyTorch 训练流程详解

## 项目概述

**OpenPI** 是 Physical Intelligence 团队开源的机器人视觉-语言-动作模型（VLA）项目，核心模型包括：
- **π₀** — Flow-based VLA
- **π₀-FAST** — Autoregressive VLA
- **π₀.₅** — 升级版 π₀，通过知识隔离提升泛化

## π₀ 计算图全貌

### 两个核心入口

| 方法 | 用途 | 模式 |
|------|------|------|
| `forward()` | **训练** — 计算 Flow Matching 损失 | 单次前向 |
| `sample_actions()` | **推理** — 采样动作 | 循环去噪（10步 Euler） |

---

## 一、训练前向（forward）

```
Observation
    │
    ├── images ──→ SigLIP (embed_image) ──┐
    ├── lang_tokens ──→ Gemma Token Embed ─┤── concat ──→ PaliGemma ──→ actions
    ├── state ──→ state_proj ─────────────┤         │
    └── actions + noise + timestep ────────┘      (action_horizon)
                                                    │
                                              action_out_proj
                                                    │
                                              MSE Loss vs (noise - actions)
```

**核心公式**（Flow Matching）:
```python
time_expanded = time[:, None, None]
x_t = time_expanded * noise + (1 - time_expanded) * actions  # 线性插值
u_t = noise - actions  # 目标向量场
v_t = model.predict(x_t, timestep)  # 预测向量场
loss = MSE(u_t, v_t)
```

---

## 二、推理采样（sample_actions）

推理是 **10 步 Euler 去噪循环**：

```
noise (随机) ──→ [去噪循环 10 次] ──→ 干净动作
                        │
                        ↓
              denoise_step(x_t, timestep)
                    │
                    ├─ embed_suffix(state, x_t, timestep)  ← 新计算
                    ├─ past_key_values (来自 prefix 的 KV Cache)
                    └─ PaliGemma.forward(inputs_embeds=[None, suffix_embs])
                              │
                              └─ action_out_proj → v_t
                                    │
                              x_{t-1} = x_t + dt * v_t  (Euler step)
```

---

## 三、Prefix 和 Suffix

### Prefix（前缀）— 可attend到所有人

```
┌─────────────────────────────────────────────────┐
│  image_tokens_0 (相机0)  ─┐                     │
│  image_tokens_1 (相机1)  ─┼──→ 互相 attend ───→ output
│  image_tokens_2 (相机2)  ─┘                     │
│                           │                      │
│  lang_tokens (语言指令)  ──────────────────────→ │
└─────────────────────────────────────────────────┘
```

在 π₀ 中，**prefix = 视觉观测 + 文本指令**。

### Suffix（后缀）— 只能attend到prefix、不能被prefix看到

```
┌──────────────────────────┬────────────────────────────┐
│       PREFIX             │         SUFFIX              │
│  (image + lang tokens)   │    (state + action)        │
│                          │                            │
│  ✅ 可以attend到prefix   │  ❌ 不能attend到自己        │
│  ✅ 可以attend到suffix   │  ✅ 可以attend到prefix     │
└──────────────────────────┴────────────────────────────┘
```

在 π₀ 中，**suffix = state（机器人状态）+ action tokens（动作序列）**。

---

## 四、π₀ vs π₀.₅ 结构差异

```
                    π₀                           π₀.₅
              ┌──────────────┐            ┌──────────────┐
  state ─────→│  state_proj  │            │  (融入lang)  │
              └──────┬───────┘            └──────────────┘
                     │ concat                      │
  action ──→ action_in_proj ─┐                    │
                              │                    │
  time ──→ sin/cos ─→ time_mlp_in ─┐               │
                                   │               │
              action_time_mlp_in ←┘               │
                                   │               │
              action_time_mlp_out ←────────────────┘
                     │
              给 Standard RMSNorm
```

---

## π₀ PyTorch 训练完整流程

### 入口：`scripts/train_pytorch.py`

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_pytorch.py pi05_libero --exp_name my_exp
```

### 流程概览

```
scripts/train_pytorch.py::train_loop()
         │
         ├─ 1. setup_ddp()           # DDP 多卡初始化
         ├─ 2. build_datasets()      # 构建数据加载器
         ├─ 3. 创建 PI0Pytorch 模型  # 模型实例化
         ├─ 4. 创建 AdamW 优化器     # 优化器
         ├─ 5. 训练循环
         │       │
         │       ├─ loader 迭代
         │       ├─ model(observation, actions)  ← 前向
         │       ├─ loss.backward()              ← 反向
         │       ├─ clip_grad_norm_
         │       └─ optim.step()
         │
         └─ 6. save_checkpoint()      # 保存检查点
```

---

## 第1步：初始化 (`setup_ddp`)

```python
# train_pytorch.py:94
def setup_ddp():
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
```

---

## 第2步：构建数据集 (`build_datasets`)

```python
# train_pytorch.py:125
def build_datasets(config):
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()
```

### 数据加载链路

```
create_data_loader(config)
         │
         ├─ config.data.create()  → LeRobotDataConfig
         │
         ├─ create_torch_dataset(data_config)
         │     │
         │     ├─ LeRobotDataset(repo_id)  ← 从 HuggingFace 加载
         │     │
         │     └─ TransformedDataset  (加入 PromptFromLeRobotTask transform)
         │
         └─ transform_dataset(dataset, data_config)
               │
               ├─ repack_transforms   (解包 LeRobot 格式)
               ├─ data_transforms     (归一化 state/action)
               └─ NormalizeTransform  (用 norm_stats 归一化)
```

### 最终输出

```python
observation = {
    "image":     {"base_0_rgb": Tensor[B,3,224,224], ...},
    "image_mask": {"base_0_rgb": Tensor[B], ...},
    "state":     Tensor[B, 32],         # 归一化后的末端位姿
    "tokenized_prompt": Tensor[B, 128], # "fold the towel"
    "tokenized_prompt_mask": Tensor[B, 128],
}
actions = Tensor[B, 50, 32]  # 50步动作，每步32维
```

---

## 第3步：创建模型

```python
# train_pytorch.py:409
model_cfg = openpi.models.pi0_config.Pi0Config(
    dtype="bfloat16",
    action_dim=32,
    action_horizon=50,
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_300m",
    pi05=True,  # ← 关键！区分 π₀ 和 π₀.₅
)

model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
model.gradient_checkpointing_enable()  # 节省显存
```

### 模型层级结构

```
PI0Pytorch
    │
    ├─ paligemma_with_expert: PaliGemmaWithExpertModel
    │     │
    │     ├─ paligemma: PaliGemma (VLM)
    │     │     ├─ vision_tower: SigLIP (ViT So400m/14)
    │     │     └─ language_model: GemmaDecoder (2B params)
    │     │
    │     └─ gemma_expert: ExpertGemma (300M params, action expert)
    │           └─ model: GemmaDecoder (共享注意力, 独立MLP)
    │
    ├─ action_in_proj: Linear(32 → 2048)
    ├─ action_out_proj: Linear(2048 → 32)
    │
    └─ π₀ 模式:
    │     ├─ state_proj: Linear(32 → 2048)
    │     ├─ action_time_mlp_in: Linear(4096 → 2048)
    │     └─ action_time_mlp_out: Linear(2048 → 2048)
    │
    └─ π₀.₅ 模式:
          ├─ time_mlp_in: Linear(2048 → 2048)
          └─ time_mlp_out: Linear(2048 → 2048)
```

---

## 第4步：优化器

```python
# train_pytorch.py:458
optim = torch.optim.AdamW(
    model.parameters(),
    lr=peak_lr,           # e.g. 1e-4
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1,
)

# 学习率调度：cosine decay with warmup
def lr_schedule(step):
    if step < warmup_steps:
        return linear_warmup(step)
    progress = (step - warmup_steps) / decay_steps
    return end_lr + (peak_lr - end_lr) * 0.5 * (1 + cos(π * progress))
```

---

## 第5步：训练循环

```python
# train_pytorch.py:509
while global_step < num_train_steps:
    for observation, actions in loader:
        # ===== 5.1 数据移到 GPU =====
        observation = jax.tree.map(lambda x: x.to(device), observation)
        actions = actions.to(device)

        # ===== 5.2 学习率更新 =====
        for pg in optim.param_groups:
            pg["lr"] = lr_schedule(global_step)

        # ===== 5.3 前向传播 =====
        losses = model(observation, actions)  # ← forward()
        loss = losses.mean()

        # ===== 5.4 反向传播 =====
        loss.backward()

        # ===== 5.5 梯度裁剪 =====
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.optimizer.clip_gradient_norm
        )

        # ===== 5.6 优化器更新 =====
        optim.step()
        optim.zero_grad(set_to_none=True)

        # ===== 5.7 日志 & 保存 =====
        if global_step % log_interval == 0:
            logging.info(f"step={global_step} loss={loss.item()} ...")
        if global_step % save_interval == 0:
            save_checkpoint(model, optim, global_step, config, ...)
```

---

## 核心：前向传播 (`PI0Pytorch.forward`)

### forward 完整代码追踪

#### 第1步：预处理观测

```python
# pi0_pytorch.py:319
images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
```

```
输入 Observation {
    images: {"base_0_rgb": Tensor[B,224,224,3], ...}   # 多路相机
    image_masks: {"base_0_rgb": Bool, ...}
    state: Tensor[B, action_dim]                     # 机器人末端位姿等
    tokenized_prompt: Tensor[B, max_token_len]       # 分词后的指令
    tokenized_prompt_mask: Tensor[B, max_token_len]
}
         │
         ↓ preprocess_observation_pytorch()
         │
images: [Tensor[B,224,224,3] × 3路相机]   # 256个patch，每个1024维
img_masks: [Tensor[B] × 3]
lang_tokens: Tensor[B, max_token_len]
lang_masks: Tensor[B, max_token_len]
state: Tensor[B, action_dim]
```

#### 第2步：采样噪声和时间

```python
# pi0_pytorch.py:321-329
noise = self.sample_noise(actions.shape, actions.device)
time = self.sample_time(actions.shape[0], actions.device)

time_expanded = time[:, None, None]
x_t = time_expanded * noise + (1 - time_expanded) * actions   # 线性插值
u_t = noise - actions                                          # 目标向量
```

```
actions:     [B, action_horizon, action_dim]  例如 [B, 50, 32]
noise:       [B, 50, 32]  ~ N(0,1)

time:        [B]  ∈ (0,1)  — Beta(1.5,1.0) 采样，偏向中间值

x_t = t * noise + (1-t) * actions
     含义：time=1时完全噪声，time=0时完全干净动作

u_t = noise - actions
     含义：Flow Matching 目标向量场
```

#### 第3步：构建 Prefix Embeddings

```python
# pi0_pytorch.py:331
prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
```

```python
# embed_prefix 内部
for img, img_mask in zip(images, img_masks):
    img_emb = self.paligemma_with_expert.embed_image(img)
    # embed_image = SigLIP vision_tower
    # 输出: [B, 256, 1024]  (256个image tokens, 1024维)
    embs.append(img_emb)
    pad_masks.append(img_mask[:, None].expand(bsize, 256))
    att_masks += [0] * 256  # 0 = 可以attend到任何人

lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
# embed_language = Gemma token embedding
# 输出: [B, max_token_len, 2048]
embs.append(lang_emb)
att_masks += [0] * max_token_len

# 最终:
prefix_embs:    [B, 256*3 + max_token_len, 2048]
prefix_pad_masks: [B, 256*3 + max_token_len]  # 1=有效token
prefix_att_masks:  [B, 256*3 + max_token_len]  # 0=可以attend
```

#### 第4步：构建 Suffix Embeddings

```python
# pi0_pytorch.py:332
suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
```

π₀ 模式（`config.pi05=False`）:
```python
# state → state_proj
state_emb = self.state_proj(state)           # [B, action_dim] → [B, 1, 2048]

# x_t → action_in_proj
action_emb = self.action_in_proj(x_t)         # [B, 50, 32] → [B, 50, 2048]

# time → sin/cos positional embedding
time_emb = create_sinusoidal_pos_embedding(time, 2048, ...)
                                        # [B, 2048]

# 拼接 + MLP
action_time_emb = torch.cat([action_emb, time_emb[:, None, :].expand_as(action_emb)], dim=-1)
                                     # [B, 50, 4096]
action_time_emb = self.action_time_mlp_in(action_time_emb)  # SiLU
action_time_emb = self.action_time_mlp_out(action_time_emb)  # [B, 50, 2048]

# 最终 suffix = state_emb + action_time_emb
suffix_embs: [B, 1 + action_horizon, 2048] = [B, 51, 2048]

# attention masks
att_masks = [1]              # state token: 不能被prefix看见
         + [1] + [0]*49     # action tokens: 第0个不能被prefix看见(因果边界)
```

π₀.₅ 模式（`config.pi05=True`）:
```python
# 关键区别：state 不作为独立token，而是融入 language
action_emb = self.action_in_proj(x_t)       # [B, 50, 32] → [B, 50, 2048]
time_emb = self.time_mlp_in(time_emb)       # [B, 2048] → [B, 2048] (SiLU)
time_emb = self.time_mlp_out(time_emb)     # [B, 2048] → [B, 2048] (SiLU)
time_emb = F.silu(time_emb)

# adarms_cond = time_emb  ← 关键！作为 adaRMSNorm 的条件输入
adrms_cond = time_emb   # [B, 2048]

suffix_embs: [B, action_horizon, 2048] = [B, 50, 2048]  (没有state token)
```

#### 第5步：拼接并构建注意力掩码

```python
# pi0_pytorch.py:340-347
pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)  # [B, 307+51]
att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)  # [B, 307+51]

att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
position_ids = torch.cumsum(pad_masks, dim=1) - 1
att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
```

```
完整序列结构:
┌─────────────── 307 tokens ────────────────┬──── 51 tokens (suffix) ─────┐
│  img0(256) │ img1(256) │ lang(40) │ ... │ state │ action_0 │ ... │ action_49 │
└───────────────────────────────────────────┴──────────────────────────────┘
          PREFIX                                        SUFFIX

att_masks:
  img: 0=全连接  │  lang: 0=全连接  │  state: 1(单向)  │  actions: 第0个=1，后续=0
```

#### 第6步：通过 PaliGemmaBlock

```python
# pi0_pytorch.py:351-358
(_, suffix_out), _ = self.paligemma_with_expert.forward(
    attention_mask=att_2d_masks_4d,
    position_ids=position_ids,
    past_key_values=None,
    inputs_embeds=[prefix_embs, suffix_embs],  # 两个输入！
    use_cache=False,
    adarms_cond=[None, adarms_cond],           # prefix无条件，suffix有time条件
)
```

`PaliGemmaWithExpertModel.forward` 内部的核心循环：

```python
# gemma_pytorch.py:156-255  compute_layer_complete

# 第 i 层：
for i, (paligemma_layer, expert_layer) in enumerate(layers):
    # 1. QKV 投影 + 分头
    q0 = paligemma_layer.self_attn.q_proj(hidden_state_0)  # prefix
    q1 = expert_layer.self_attn.q_proj(hidden_state_1)     # suffix

    # 2. 拼接 QKV
    Q = cat([q0, q1], dim=2)  # 跨模型共享attention！
    K = cat([k0, k1], dim=2)
    V = cat([v0, v1], dim=2)

    # 3. RoPE + Eager Attention
    Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
    att_output = eager_attention(Q, K, V, mask)

    # 4. 分离 + O投影 + 残差
    out_0, out_1 = split(att_output)
    out_0 = paligemma_layer.o_proj(out_0) + hidden_state_0  # 残差
    out_1 = expert_layer.o_proj(out_1) + hidden_state_1

    # 5. AdaRMSNorm (π₀.₅) 或 标准RMSNorm (π₀)
    out_0 = paligemma_layer.post_attention_layernorm(out_0, cond=None)
    out_1 = expert_layer.post_attention_layernorm(out_1, cond=adrms_cond)  # ← 关键！
```

**核心思想**：
- Prefix 和 Suffix **共享同一套 Transformer 层**
- 区别在于 `cond` 参数：`prefix` 无条件，`suffix` 有 time 条件
- 这就是为什么叫 "PaliGemma **With Expert**" — Expert 层通过 adaRMSNorm 接收 time 信息

#### 第7步：输出投影 + Loss

```python
# pi0_pytorch.py:365-373
suffix_out = suffix_out[:, -self.config.action_horizon:]  # 取最后action_horizon个token
v_t = self.action_out_proj(suffix_out)                    # [B, 50, 2048] → [B, 50, 32]

return F.mse_loss(u_t, v_t, reduction="none")  # MSE between (noise-actions) and predicted
```

```
v_t: [B, 50, 32]  — 预测的向量场
u_t: [B, 50, 32]  — 目标向量场 (noise - actions)

loss = MSE(u_t, v_t)  — 让模型学会从噪声预测目标向量
```

---

## 单步计算图

```
Batch (observation, actions)
         │
         ├── obs.state:        [B, 32]
         ├── obs.images:       [B, 3, 224, 224] × 3相机
         ├── obs.lang_tokens:  [B, 128]
         └── actions:          [B, 50, 32]
              │
              ├── sample_noise() → [B, 50, 32]  ~ N(0,1)
              ├── sample_time()  → [B]          ~ Beta(1.5,1.0) ∈ (0,1)
              │
              ├── x_t = t*noise + (1-t)*actions
              ├── u_t = noise - actions
              │
              ├── embed_prefix(images, lang)
              │     SigLIP.vision_tower   → [B, 768, 1024]
              │     Gemma.token_embedding  → [B, 128, 2048]
              │     concat → [B, prefix_len, 2048]
              │
              ├── embed_suffix(state, x_t, t)
              │     π₀.₅: action_in_proj + time_mlp → adarms_cond
              │     concat → [B, 50, 2048]
              │
              ├── concat([prefix, suffix]) → [B, total_len, 2048]
              │
              ├── PaliGemmaWithExpert.forward()
              │     for each layer:
              │       QKV_proj → RoPE → EagerAttention → O_proj
              │       → GatedResidual → adaRMSNorm(cond=time)
              │     final_norm
              │
              ├── 取最后50个token → action_out_proj → [B, 50, 32]
              │
              └── MSE(u_t, v_t) → scalar loss
```

---

## Flow Matching 直观理解

```
t=0 (干净动作)          t=0.5 (混合)           t=1 (纯噪声)
    🎬                    🔄                    📺

actions            = 0.5*noise + 0.5*actions
u_t = noise - actions  (恒定目标)

模型学习: 给定 x_t 和 t，预测 u_t
推理时: 从纯噪声开始，迭代去噪 t=1→0
```

---

## 关键类和方法一览

| 文件 | 类/方法 | 作用 |
|------|---------|------|
| `train_pytorch.py:309` | `train_loop()` | 训练入口 |
| `train_pytorch.py:125` | `build_datasets()` | 构建数据加载器 |
| `train_pytorch.py:409` | `PI0Pytorch(model_cfg)` | 模型实例化 |
| `pi0_pytorch.py:317` | `PI0Pytorch.forward()` | **训练前向 + Loss计算** |
| `pi0_pytorch.py:377` | `PI0Pytorch.sample_actions()` | 推理采样 |
| `pi0_pytorch.py:187` | `embed_prefix()` | 视觉+语言 embedding |
| `pi0_pytorch.py:238` | `embed_suffix()` | 状态+动作 embedding |
| `gemma_pytorch.py:90` | `PaliGemmaWithExpertModel.forward()` | 共享 Transformer |
| `gemma_pytorch.py:49` | `GemmaRMSNorm` | adaRMSNorm 实现 |
| `preprocessing_pytorch.py:20` | `preprocess_observation_pytorch()` | 图像预处理+增强 |
| `data_loader.py:130` | `create_torch_dataset()` | LeRobot 数据集封装 |
| `data_loader.py:172` | `transform_dataset()` | 数据变换 pipeline |
| `data_loader.py:53` | `TransformedDataset` | 变换数据集容器 |

---

## 推理采样流程（sample_actions）

推理使用 **10 步 Euler 去噪**，与训练前向的主要区别：

| | 训练 (forward) | 推理 (sample_actions) |
|--|--|--|
| **Prefix 处理** | 每步重新计算 | **只算一次**，缓存 KV |
| **Suffix 处理** | 一次性输入全部 action tokens | **每步去噪一次** |
| **Attention** | Full prefix + suffix | prefix 用 cache，suffix 逐token生成 |

### sample_actions 代码

```python
def sample_actions(self, device, observation, noise=None, num_steps=10):
    bsize = observation.state.shape[0]

    # 1. 预处理观测
    images, img_masks, lang_tokens, lang_masks, state = \
        self._preprocess_observation(observation, train=False)

    # 2. 只计算一次 prefix（使用 KV Cache）
    prefix_embs, prefix_pad_masks, prefix_att_masks = \
        self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

    # 3. 第一次 forward：计算 prefix 的 KV Cache
    _, past_key_values = self.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    # 4. Euler 去噪循环
    dt = -1.0 / num_steps  # dt = -0.1
    x_t = noise  # 初始为纯噪声

    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    while time >= -dt / 2:
        expanded_time = time.expand(bsize)
        v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
        x_t = x_t + dt * v_t  # Euler 步
        time += dt

    return x_t
```

### denoise_step

```python
def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
    # 1. 每次重新计算 suffix embedding（x_t 在变化）
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = \
        self.embed_suffix(state, x_t, timestep)

    # 2. 构建跨 prefix-suffix 的注意力掩码
    full_att_2d_masks = ...

    # 3. 使用缓存的 prefix KV，只计算 suffix 部分
    outputs_embeds, _ = self.paligemma_with_expert.forward(
        attention_mask=full_att_2d_masks_4d,
        position_ids=position_ids,
        past_key_values=past_key_values,  # 传入缓存的 KV
        inputs_embeds=[None, suffix_embs],  # 只传入 suffix
        use_cache=False,
        adarms_cond=[None, adarms_cond],
    )

    # 4. 输出投影
    suffix_out = outputs_embeds[1][:, -self.config.action_horizon:]
    return self.action_out_proj(suffix_out)
```

---

## KV Cache 优化

`sample_actions` 中，prefix 的 KV 被缓存：

```python
# 第一次 forward
_, past_key_values = self.paligemma_with_expert.forward(
    inputs_embeds=[prefix_embs, None],  # 只用 prefix
    use_cache=True,
)
# 后续 denoise_step 传入 past_key_values，避免重复计算 prefix
```

这就是为什么 `embed_prefix` **只在推理开始时调用一次**，而 `embed_suffix` 需要在**每步去噪中重新计算**（因为动作 x_t 在变化）。
