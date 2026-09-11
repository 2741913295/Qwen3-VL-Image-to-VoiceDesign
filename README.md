# Qwen3-VL Image-to-VoiceDesign

这是一个基于 Qwen3-VL 与 Qwen3-TTS VoiceDesign 的角色音色设计项目。输入人物图片后，系统先生成可解释、可区分的中文声线描述，再将该描述作为 `instruct` 交给 Qwen3-TTS 生成对应音频。

项目面向“角色声音设计”，不会声称预测真人的真实声音、职业或人格。图片中的年龄感、面部成熟度、体型、造型、姿态和整体视觉气质，只用于设计一套与角色形象合理匹配的虚构声线。

## 处理链路

```text
人物图片
  → Visual Profile（视觉特征）
  → Persona Profile（虚构角色人设）
  → Speaking Behavior（说话习惯）
  → Voice Fingerprint（声线指纹）
  → 中文 VoiceDesign instruct
  → Qwen3-TTS 音频
```

图转文阶段不是让模型自由输出一段笼统描述，而是先完成多阶段结构化分析，再由代码组合为最终自然语言 instruct：

- Visual Profile：年龄感、性别呈现、面部成熟度、体型和视觉重量、造型结构、姿态、表情及角色气质。
- Persona Profile：虚构身份、核心性格、角色原型、价值倾向、行动动机、社交方式及情绪模式。
- Speaking Behavior：语速、停顿习惯、句子节奏、咬字方式、情绪反应、句尾走向及声音距离感。
- Voice Fingerprint：音调中心、音高变化、声音厚度、明暗与冷暖、共鸣位置和深度、气声、粗糙度、鼻音、起声、张力、咬字及动态范围。

批量处理时会同时比较人物人设、说话行为和核心声学属性。当两个角色过于接近时，系统会重新设计或分配差异化声线，减少大量人物收敛到同一套“年轻女性清亮轻柔”或“成熟男性低沉威严”模板的问题。

## 主要功能

- 单张图片生成 Qwen3-TTS 可直接使用的中文 instruct。
- 单张图片生成 instruct 和 WAV 音频。
- 文件夹批量处理，支持中英文文件名和常见图片格式。
- Qwen3-VL 与 Qwen3-TTS 模型常驻，批量任务只加载一次模型。
- 支持 vLLM 0.28.x 与 vLLM-Omni 0.28.x。
- 支持 Transformers 作为 Qwen3-VL 的兼容回退后端。
- 输出文件名与输入图片名保持一致。
- 支持可选表达约束，例如愤怒、悲伤、温柔、语速和力度要求。
- 使用批量声线注册文件进行跨图片相似度检测和音色去重。

## 项目结构

```text
.
├── run_voice_design.py              # 图片 → instruct
├── run_image_to_audio.py            # 单图 → instruct + WAV
├── run_batch_image_to_audio.py      # 文件夹批量生成
├── voice_design/
│   ├── inference.py                 # Qwen3-VL 分阶段推理
│   ├── prompts.py                   # 视觉、人设、行为和声线 Prompt
│   ├── profiles.py                  # 结构校验、差异化与文案组合
│   ├── tts_inference.py             # vLLM-Omni TTS 推理
│   └── qwen3_tts_low_memory.yaml    # TTS 低显存配置
└── tests/                            # 核心逻辑测试
```

本仓库不包含模型权重、输入图片、输出文本、音频、运行日志、密钥以及 vLLM/vLLM-Omni 源码副本。

## 环境准备

建议使用独立 Conda 环境，并根据服务器 CUDA 与驱动版本安装相匹配的 PyTorch、vLLM 0.28.x 和 vLLM-Omni 0.28.x。

```bash
conda create -n qwen3vl python=3.12 -y
conda activate qwen3vl

pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/simple
pip install -r requirements_voice_design.txt
```

如果使用 vLLM 推理，还需要安装同一版本系列的运行框架：

```bash
pip install -e /path/to/vllm-0.28.0
pip install -e /path/to/vllm-omni-0.28.0
```

实际安装方式应以 vLLM 和 vLLM-Omni 官方针对当前 CUDA 环境的说明为准。项目会检查安装包与指定源码目录是否同为 `0.28.x`。

## 模型准备

模型需要提前下载到服务器，本仓库不会自动联网下载：

- Qwen3-VL：`Qwen3-VL-8B-Instruct` 或兼容的 FP8 权重。
- Qwen3-TTS：`Qwen3-TTS-12Hz-1.7B-VoiceDesign`。

运行时通过参数传入本地模型路径，不要把权重复制进 Git 仓库。

## 使用方法

### 1. 只生成声线 instruct

```bash
python run_voice_design.py \
  --image "/path/to/data/character.png" \
  --model-path "/path/to/Qwen3-VL-8B-Instruct-FP8" \
  --backend vllm \
  --vllm-source-path "/path/to/vllm-0.28.0" \
  --output-dir "/path/to/output"
```

终端会打印最终 instruct，同时保存：

```text
/path/to/output/character.txt
```

### 2. 图片生成 instruct 和音频

```bash
python run_image_to_audio.py \
  --image "/path/to/data/character.png" \
  --text "今天的天气比想象中安静，我刚从外面回来。" \
  --model-path "/path/to/Qwen3-VL-8B-Instruct-FP8" \
  --vllm-source-path "/path/to/vllm-0.28.0" \
  --tts-model-path "/path/to/Qwen3-TTS-12Hz-1.7B-VoiceDesign" \
  --vllm-omni-source-path "/path/to/vllm-omni-0.28.0" \
  --tts-deploy-config "voice_design/qwen3_tts_low_memory.yaml" \
  --output-dir "/path/to/output"
```

输出结果：

```text
/path/to/output/character.txt
/path/to/output/character.wav
```

### 3. 增加表达约束

```bash
python run_image_to_audio.py \
  --image "/path/to/data/character.png" \
  --text "这件事情我已经忍耐很久了。" \
  --instruction "使用非常愤怒的语气说话" \
  --model-path "/path/to/Qwen3-VL-8B-Instruct-FP8" \
  --vllm-source-path "/path/to/vllm-0.28.0" \
  --tts-model-path "/path/to/Qwen3-TTS-12Hz-1.7B-VoiceDesign" \
  --vllm-omni-source-path "/path/to/vllm-omni-0.28.0"
```

普通情绪指令只修改当前表达的情绪、语速、力度和韵律，不会随意改变图片确定的基础声线。只有当指令明确要求年龄、性别或音色身份时，才会应用对应的声线覆盖。

### 4. 批量处理文件夹

```bash
python run_batch_image_to_audio.py \
  --input-dir "/path/to/data/test" \
  --output-dir "/path/to/output/test" \
  --text "今天的天气比想象中安静，我刚从外面回来。" \
  --model-path "/path/to/Qwen3-VL-8B-Instruct-FP8" \
  --vllm-source-path "/path/to/vllm-0.28.0" \
  --tts-model-path "/path/to/Qwen3-TTS-12Hz-1.7B-VoiceDesign" \
  --vllm-omni-source-path "/path/to/vllm-omni-0.28.0" \
  --tts-deploy-config "voice_design/qwen3_tts_low_memory.yaml" \
  --skip-existing
```

批处理会生成同名 TXT 和 WAV，并在输出目录保存隐藏的声线注册文件，用于后续人物之间的差异化比较。删除输出目录后重新运行，会重新建立整批人物的声线组合。

## Transformers 回退

若暂时不使用 vLLM，可让 Qwen3-VL 使用 Transformers：

```bash
python run_voice_design.py \
  --image "/path/to/character.png" \
  --model-path "/path/to/Qwen3-VL-8B-Instruct" \
  --backend transformers \
  --attn-implementation sdpa
```

Transformers 回退仅影响图片分析阶段，不改变后续 Persona、Speaking Behavior、Voice Fingerprint 和 instruct 组合规则。

## 输出示例

```text
角色定位为边境守备军的老练指挥官，性格克制、警觉而果决。40岁左右的成熟男性声线，音调偏低且变化范围较窄，声体厚实、明亮度偏暗，以深层胸腔共鸣为主，带轻微颗粒感；起声坚定，声音张力较高，咬字短促清晰，语速不急，停顿有控制感，句尾下沉收束，整体听感沉着、有压迫感但不过分表演化。
```

最终 TXT 只保留对角色声音和说话方式有意义的描述，不直接输出服装、发型或图片分析过程。

## 数据与安全

- 不要提交真实业务图片、生成音频、输出 TXT 或日志。
- 不要在代码中写入 API Key、访问令牌、账号或服务器密码。
- 模型权重与推理框架源码应在部署环境中单独管理。
- `.gitignore` 已默认排除常见模型格式、数据目录、输出目录、音频、缓存、环境文件和本地推理框架副本。

## 说明

本项目基于 Qwen3-VL 官方代码与 Qwen3-TTS VoiceDesign 能力进行二次开发，重点工作集中在人物视觉到角色人设、说话行为和声线指纹的映射，以及批量人物之间的音色差异化约束。模型能力与许可请同时参考对应上游项目。

## License

沿用项目根目录中的 `LICENSE`。
