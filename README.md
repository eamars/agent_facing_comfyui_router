# ComfyUI 本地图像生成服务

一个面向个人使用和 AI Agent 的本地图像生成服务。通过 ComfyUI 网页或统一 HTTP API，使用不同模型完成文生图、图生图、指令编辑、局部重绘、人物参考和姿态控制。

项目提供可查询的模型与工作流目录。调用方可以指定工作流，也可以按题材、风格和输入类型选择路由，无需自行拼装 ComfyUI 节点。模型在任务之间自动热切换，不需要重启服务。

## 能做什么

- **图像生成**：使用 Qwen-Image、MOSS-Image、Anima、Flux.2/Klein 和 SDXL/Illustrious 系列模型生成写实图像、动漫人物和插画。
- **布局控制**：通过 MOSS-Image 的布局 JSON 指定文字和物体在画面中的位置与内容。
- **参考图编辑**：通过 Qwen-Image-2.1 输入一张图片和自然语言指令，修改颜色、服装、场景等内容。
- **扩图**：同一 Qwen 编辑路由按像素向四周扩展图片，新区域按提示词补全，原图像素保持不变。
- **裁剪与缩放**：不经模型、不进队列，即时裁剪或缩放上传图片和生成结果，结果可直接作为下一次输入。
- **超分辨率放大**：用放大模型把图片放大到 1–4 倍（动漫与照片各一个模型），约 3 秒，不触发模型切换。
- **图生图与局部重绘**：使用参考图生成变体，或提供蒙版重绘指定区域并保留蒙版外的原图像素。
- **人物参考**：通过 IPAdapter 参考人物外观。
- **姿势**：同一个 `pose` 参数接受姿势库编号或上传的图片，服务提取骨架后按模型家族接入控制；目前 SDXL / Illustrious、Anima、Qwen-Image 文生图和 Moody Desire（Flux.2 Klein）路由已支持。
- **自动选路**：根据题材、画风和输入方式，从可用工作流中选择合适的模型组合。
- **队列与热切换**：串行处理生成任务，同一工作流复用模型缓存，切换工作流时自动释放并加载所需模型。
- **Agent 接入**：提供能力发现、参数说明、图片上传、任务状态和结果下载接口，也保留原生 ComfyUI 接口。

当前部署面向 Windows、Python 3.12 和 RTX 4090。启动器锁定本机指定的 4090，支持 RAM offload 和 CPU VAE；模型权重与生成过程均在本机运行。

## 支持的模型

下表列出可通过命名工作流调用的模型组合。图生图指基于参考图的重采样；Qwen 的指令编辑使用独立工作流。

| 模型 / 组合 | 主要用途 | 支持方式 | 工作流 ID |
| --- | --- | --- | --- |
| Qwen-Image-2.1 Turbo · 官方 BF16 | 自然语言生成、图中文字、参考图指令编辑 | 文生图、单图编辑 | `qwen-image-2.1-t2i`、`qwen-image-2.1-edit` |
| MOSS-Image-SFT · 官方 BF16 | 布局 JSON 精确排版、长提示词、图中中英文文字 | 文生图（自然语言 / 标签 / 布局 JSON） | `moss-image-sft-t2i` |
| Anima Turbo v1.1 · 官方 BF16 | 通用动漫、人物与插画 | 文生图、图生图 | `anima-turbo-v1-1` |
| Moody Anima Mix v1.0 | 低步数动漫生成 | 文生图、图生图 | `moody-anima-mix` |
| Moody Desire Mix v3.0 · Flux.2/Klein | 写实人像为主（擅长成人写实图），也能画动漫 | 文生图、图生图 | `moody-desire-v30` |
| Animagine XL 3.1 + NIKKE LoRA | NIKKE 风格；可选 Rapi 角色适配 | 文生图、图生图 | `animagine-xl-3.1-nsfw` |
| JANKU v7.77 · Illustrious/NoobAI + Chenkin 系 | 主流动画风：线条较粗、背景细致、电影感光影，默认脸型偏成熟 | 文生图、图生图 | `janku-illustrious` |
| JANKU v5.0 · Illustrious/NoobAI + RouWei 系 | 视觉小说（Galgame）风：脸型柔和精致、线条细、浅色光泽与腮红，画师标签最强 | 文生图、图生图 | `janku-v5-illustrious` |
| One Obsession v24 · Illustrious 系 | 纹理、戏剧性光影与 2.5D 风格 | 文生图、图生图 | `one-obsession` |
| Anima Turbo v1.1 + DeepSeek 鲸鱼娘 LoRA（ZipZipPipe 画风） | 大肥鱼女仆装形象 | 文生图、图生图 | `deepseek-whale-girl-anima-turbo` |
| Anima Turbo + Claude 娘 LoRA | Claude 娘（ors scholar，橙发白裙） | 文生图、图生图 | `claude-girl-anima-turbo` |
| Anima Turbo + ChatGPT 龙娘 LoRA | ChatGPT 龙娘（白发龙角、龙翼龙尾） | 文生图、图生图 | `chatgpt-girl-anima-turbo` |
| Anima Turbo + 千问娘 LoRA | 千问娘（紫蓝长发侧辫、中式长袍） | 文生图、图生图 | `qwen-girl-anima-turbo` |
| Anima Turbo + Gemini 娘 LoRA | Gemini 娘 / 哈基米（渐变发双丸子、猫耳异色瞳） | 文生图、图生图 | `gemini-girl-anima-turbo` |
| Anima Turbo + aigirls 多角色 LoRA | GLM 智谱娘（黑发狐狸娘、黑白哥特洛丽塔）、Grok 娘（金发双马尾吸血鬼）、Kimi 娘（银蓝卷发、星空斗篷） | 文生图、图生图 | `glm-girl-anima-turbo`、`grok-girl-anima-turbo`、`kimi-girl-anima-turbo` |

路由不区分 SFW 与 NSFW：画面内容由提示词决定，各模型家族都能生成成人内容（MOSS-Image 未测试），路由按题材和画风选择。旧的选路参数 `content_mode`、`input_type` 和 `prompt_granularity` 仍被接受但不起作用，响应中会返回 `deprecated_selector` 警告。

### 局部重绘与人物参考

| 工作流 ID | 模型 / 组件 | 用途 | 除提示词外的必需输入 |
| --- | --- | --- | --- |
| `anima-turbo-inpaint` | Anima Turbo v1.1 | 蒙版区域重绘 | `input_image`、`mask_image` |
| `moody-anima-inpaint` | Moody Anima Mix | 蒙版区域重绘 | `input_image`、`mask_image` |
| `anima-turbo-face-reference-ipadapter` | Anima Turbo + Character Reference IPAdapter | 人物外观参考 | `face_image` |
| `sdxl-face-reference-ipadapter` | Animagine XL 3.1 + IPAdapter Plus Face | 面部外观参考 | `face_image` |

人物参考用于引导外观，不保证身份或五官完全一致。

### 姿势

姿势不是单独的工作流，而是各路由共用的可选参数 `pose`：

- **姿势库编号**：`GET /project/poses?q=跪&people=1&angle=side` 按中英文名称搜索 196 个姿势。每个姿势最多有正面、斜侧、侧面、俯视、仰视五个视角（DWPose 认不出的视角已剔除，五个视角齐全的有 47 个），返回的视角编号（如 `full_single_P035_side`）直接作为 `pose` 传入；只传姿势编号时取正面视角，没有正面时取第一个视角。
- **上传图片**：照片、插画或线稿均可，服务会复制其中人物的身体姿势。

服务用 DWPose 把输入提取为全身骨架（身体、手、脚和脸部关键点），按内容哈希缓存；找不到人体时以 `pose_not_detected` 拒绝。骨架按输出尺寸等比绘制，画法由模型家族决定；未指定宽高时，输出采用姿势图的宽高比。`pose_strength`（0–1，SDXL 默认 0.8，Anima 默认 1.0；Qwen 和 Klein 无此参数）控制贴合程度。姿势只用于从文字生成（可同时使用人物参考 `face_image`），不能改变 `input_image` 中人物的姿势：编辑、图生图和局部重绘请求带 `pose` 会返回 `pose_unsupported`。响应中的 `pose` 对象给出实际使用的姿势、检测到的人数和 `skeleton_url`（模型实际遵循的骨架图）。人物、服装和背景仍由提示词描述。

每个模型家族有自己的接入方式，实测结果（[`scripts/verify_pose_fidelity.py`](scripts/verify_pose_fidelity.py)，PCK@0.1）记录在 catalog 的 `pose_adapters`：

- **SDXL / Illustrious**（JANKU、One Obsession、Animagine 及 SDXL 人物参考）：Xinsir OpenPose ControlNet，贴合度 0.75（不加姿势 0.34）。站姿和手臂动作贴合较好；坐、跪等姿势需要在提示词中同时写出，否则常被画成站姿。
- **Anima**（Anima Turbo、Moody Anima Mix、各 AI 娘角色及 Anima 人物参考）：Anima Control Pose Preview-2（控制 LoRA 加控制嵌入，读细线全身骨架），贴合度 0.77（不加姿势 0.28）。站姿、手臂和坐姿都能跟随，测试的侧面姿势未跟随，Moody Anima Mix 跟随较弱；会让肤色偏深、裤子偏紧。角色 LoRA 路由常保持自己的站姿，需要在提示词中写出姿势。该权重仅限非商用，生成的图片不受限制。
- **Qwen-Image 文生图**：不用 ControlNet，把细线全身骨架作为一张参考图交给 Qwen 的文本编码器，并在提示词前说明只取其姿势，贴合度 0.96（不加姿势 0.44），包括侧面跪姿，人物保持提示词描述的外观；手部偶尔留下骨架痕迹，每张约多 10 秒。Qwen 编辑路由会把姿势图直接贴进画面而不改变人物姿势，因此不接受 `pose`。
- **Moody Desire（Flux.2 Klein）**：同样不用 ControlNet，把骨架编码为参考潜变量（`ReferenceLatent`）并在提示词前说明只取其姿势。用写实人像提示词测得贴合度 0.83（不加姿势 0.44），照片风格保持不变；侧面跪姿未跟随。该模型面向写实风格，用动漫标签写提示词时姿势同样跟随（0.82），但画面会偏向插画。每张约多 1 秒。

MOSS-Image 不接受任何图片输入，无法接入姿势。向不支持姿势的路由传 `pose` 会返回 `pose_unsupported` 并列出可用路由；需要姿势时在 `/project/resolve` 中加 `input_mode: "openpose-control"`，只在支持姿势的路由中选择。原 `sdxl-openpose-controlnet` 工作流已并入 `animagine-xl-3.1-nsfw`，旧编号和旧参数 `control_image` 仍可用。

姿势库来自 [ai-pose-gallery-zh](https://github.com/anhao/ai-pose-gallery-zh)（经 [gen-ai-pose-image](https://github.com/wzc520pyfm/gen-ai-pose-image) 整理），线稿著作权归ギア所有，不得商用。仓库只保存索引 [`workflows/poses.json`](workflows/poses.json)（英文名为机器翻译）；线稿在首次使用时下载并缓存到 `ComfyUI/input/_poses/`，输入列表不显示该目录。DWPose 认不出的视角已从索引中剔除，重建索引用 [`scripts/build_pose_library.py`](scripts/build_pose_library.py)。

### 如何选择

通用动漫请求使用 Anima Turbo；指定主流动漫画风（`mainstream anime`）时使用 JANKU v7.77，指定视觉小说、Galgame 或柔和精致的角色风格时使用 JANKU v5.0。两者都能生成成人内容，区别在画风：JANKU 偏光泽柔和的同人风，Anima Turbo 偏平涂赛璐珞。写实人像的成人图可选 Moody Desire（`style: "moody desire"` 或直接用工作流编号），它偏写真风格、只需 4 步；通用照片仍由 Qwen-Image 负责。明确指定 NIKKE、AI 娘化角色或模型风格时，路由器优先匹配对应组合。AI 娘化角色的 domain 为 `deepseek`（大肥鱼）、`claude girl`（Claude娘）、`chatgpt`（ChatGPT龙娘）、`qwen girl`（千问娘）、`gemini girl`（哈基米）、`glm girl`（智谱娘）、`grok girl`（Grok娘）和 `kimi girl`（Kimi娘）；单独的 `qwen` 仍指 Qwen-Image，单独的 `claude`、`gemini`、`kimi` 也会撞上其他同名角色，不作为 domain；每个角色只有一条 Anima Turbo 路由，SFW 与 NSFW 都走它（JANKU 版角色路由的形象与公开设定不符或不如 Anima 贴近，已移除），LoRA 强度统一用 `loras.character` 调整。角色一律按成年女性描写。

使用 Qwen 时，可指定 `domain: "qwen"`，并选择 `text-to-image` 或 `image-edit`（按指令修改上传的图片）。写实照片、产品、风景、海报等非动漫请求用 `domain: "photo"`、`"product"`、`"landscape"` 或 `"poster"`，选择 Qwen-Image；需要按坐标控制文字和物体的位置与大小，或生成界面原型时，用 `domain: "layout"` 或 `"ui mockup"`，选择 MOSS-Image（提示词需写完整描述，过短会导致主体变形）。每条路由可匹配的词见 agent manifest 的 `route_table[].selectors`。局部重绘和人物参考分别使用 `inpaint`、`face-reference`；需要姿势时加 `openpose-control`。

全部工作流、默认参数和输入要求见 [工作流目录](workflows/catalog.json)；模型组件、来源与文件信息见 [模型表](workflows/models.json)。运行时可查询 `GET /project/workflows` 获取实际可用状态。各项取舍的原因与实测依据见 [设计决定](docs/decisions.md)。

## 快速开始

在 PowerShell 中进入项目目录，安装运行环境并下载所需模型：

```powershell
.\scripts\setup_comfyui.ps1
.\scripts\download_models.ps1 -Profile qwen-image-2.1
.\scripts\manage_comfyui.ps1 start
```

打开 [ComfyUI 网页](http://127.0.0.1:8191)。同一可信局域网内的客户端可以通过 `http://<主机局域网 IP>:8191` 访问。

下载器支持断点续传，并校验文件大小和 SHA-256。模型按需下载，也可以使用 `-Profile all` 下载全部配置。

| 所需模型 / 功能 | 下载 Profile |
| --- | --- |
| Qwen-Image-2.1 Turbo 生成与编辑 | `qwen-image-2.1` |
| MOSS-Image-SFT 文生图与布局生成（约 27 GB） | `moss-image-sft` |
| Anima Turbo、局部重绘与 Anima 人物参考 | `anima-turbo` |
| Moody Anima Mix 与局部重绘 | `moody-anima-mix` |
| Moody Desire Mix | `moody-desire-v30` |
| Animagine XL 3.1 / NIKKE | `animagine-xl-3.1-nsfw` |
| JANKU v7.77 | `janku-illustrious` |
| JANKU v5.0 | `janku-v5-illustrious` |
| One Obsession v24 | `one-obsession` |
| DeepSeek 鲸鱼娘 LoRA（需 `CIVITAI_TOKEN`） | `deepseek-whale-girl` |
| Claude / ChatGPT / 千问 / Gemini 娘 LoRA 及 GLM / Grok / Kimi 娘共用的 aigirls LoRA（五个，需 `CIVITAI_TOKEN`） | `ai-agent-girls` |
| SDXL 人物参考附加组件 | `ipadapter-face-sdxl` |
| SDXL / Illustrious 姿势接入（Xinsir OpenPose ControlNet） | `controlnet-openpose-sdxl` |
| Anima 姿势接入（Anima Control Pose Preview-2，非商用权重） | `anima-control-pose` |
| 姿势骨架提取与人物参考的人脸裁剪（DWPose，所有姿势请求和人物参考都需要） | `pose` |

SDXL 人物参考还需要 Animagine XL 3.1 基础模型，可通过 `animagine-xl-3.1-nsfw` Profile 安装。

Qwen-Image-2.1 的两条路由使用官方 Qwen-Image-2.1-Turbo BF16 扩散模型（Comfy-Org 重新打包，8 步、CFG 1、模板中的固定采样表），配套 Qwen3VL INT8 文本编码器与 BF16 VAE（来自 SHA-256 一致的 puzhen 镜像）。基础模型本身可以生成成人内容，因此不再叠加 uncensored LoRA，也不再接受 `loras`、`steps`、`sampler_name` 和 `scheduler` 参数。4090 上 1024×1024 文生图约 33 秒，其中采样约 3 秒，其余为文本编码、CPU VAE 解码和模型搬运。原 Q4_K_M GGUF 与 LoRA 保留在本地作回滚，不再被命名工作流加载。替换的原因与依据见 [设计决定](docs/decisions.md)。

MOSS-Image-SFT 使用 OpenMOSS 发布的 5B DiT、Qwen3.5 9B 文本编码器和 FLUX.2 小解码器 VAE，按 Hugging Face 目录原样存放在 `ComfyUI/models/moss-image/MOSS-Image-SFT/`，由 `setup_comfyui.ps1` 固定版本安装的官方 MOSS-Image 节点直接读取。默认 1024×1024、50 步、Euler/sgm_uniform、CFG 3、shift 5，并在负向分支跳过第 22 层；每步计算两次；4090 上 1024×1024 约 55 秒，与 Qwen 文生图相当。

查看状态或停止服务：

```powershell
.\scripts\manage_comfyui.ps1 status
.\scripts\manage_comfyui.ps1 stop
```

默认监听 `0.0.0.0:8191`，供可信局域网使用，接口不带身份认证。仅本机使用时，以 `-BindHost 127.0.0.1` 启动；后续管理命令使用相同的 `-BindHost`。启动器按 GPU UUID 绑定本机 4090，在另一台机器部署时需要调整 [启动器](scripts/launch_comfyui.py) 中的绑定。

## 通过 API 使用

网页与 API 共用 `8191` 端口。AI Agent 可以从 `GET /.well-known/agent-manifest.json` 发现服务能力，通过 `GET /project/openapi.json` 获取接口定义。

### 选择并生成图像

如果已确定模型，直接向 `POST /project/generate` 提交工作流 ID 和提示词：

```json
{
  "workflow": "qwen-image-2.1-t2i",
  "parameters": {
    "prompt": "A blue ceramic teapot on a wooden table beside a sunlit window.",
    "seed": 921
  }
}
```

Qwen 文生图默认使用 1024×1024、8 步（Turbo 固定采样表）、Euler、CFG 1。各工作流均提供自己的默认参数，可通过工作流详情中的 `parameter_schema` 查询允许调整的字段。省略 `seed` 时会随机生成，并在响应中返回实际值。

如果希望自动选路，先调用 `POST /project/resolve`：

```json
{
  "domain": "anime",
  "input_mode": "text-to-image"
}
```

响应中的 `selected_workflow` 可用于生成请求。选路只返回工作流及其提示词建议，不会启动生成；角色、服装、动作和场景等细节写入 `parameters.prompt`。

生成响应提供 `job_id` 和 `job_url`。轮询 `job_url`，直到状态为 `succeeded` 或 `failed`；成功时通过 `outputs[].url` 获取图片。请求顶层添加 `"dry_run": true` 可检查参数和依赖而不生成图像，响应会说明工作流是否就绪及图是否通过验证。

请求超过工作流的 `max_total_pixels` 时不会被拒绝，响应 `warnings` 中会出现 `over_pixel_budget`，任务照常运行。若任务因内存或显存不足失败，`error.code` 为 `out_of_memory`。请把错误和 `GET /project/resources` 的当前空闲内存告诉请求者，由其释放内存或缩小尺寸后再重新提交。

### 用布局 JSON 生成

MOSS-Image 接受把布局 JSON 序列化后直接作为 `prompt` 字符串。坐标为整数像素，以画布中心为原点（x 向右、y 向下），`bbox` 顺序为 `[y_min, x_min, y_max, x_max]`；1024×1024 画布两轴范围均为 -512 到 512：

```json
{
  "workflow": "moss-image-sft-t2i",
  "parameters": {
    "prompt": "{\"summary\": {\"caption\": \"A minimalist poster on a cream background.\"}, \"items\": [{\"label\": \"text\", \"caption\": \"Bold title at the top.\", \"text\": \"EXPLORE\", \"bbox\": [-460, -400, -330, 400]}], \"bbox_format\": \"yxyx\", \"bbox_coord_space\": \"abs_centralized\"}"
  }
}
```

### 上传图片并编辑

先上传参考图：

```powershell
curl.exe -X POST http://127.0.0.1:8191/upload/image `
  -F "image=@.\reference.png" -F "type=input" -F "overwrite=false"
```

将返回的 `subfolder` 和 `name` 用 `/` 拼接为相对文件名，再提交编辑请求：

```json
{
  "workflow": "qwen-image-2.1-edit",
  "parameters": {
    "input_image": "reference.png",
    "prompt": "Change the blue teapot to red. Keep the background and lighting unchanged."
  }
}
```

Qwen 编辑接口支持单张参考图，输出尺寸随原图按 32 像素取整，不接受宽高或批量数量覆盖。

扩图同样使用 `qwen-image-2.1-edit`：加上 `extend`，给出每边要增加的像素（`left`、`top`、`right`、`bottom`，每边最多 4096，可省略），提示词只描述新区域应出现的内容，不需要蒙版或自行填充：

```json
{
  "workflow": "qwen-image-2.1-edit",
  "parameters": {
    "input_image": "balcony.png",
    "extend": {"left": 448, "right": 448},
    "prompt": "more of the balcony railing and the city skyline at dusk"
  }
}
```

服务把原图放在灰色画布上，在提示词前加入"灰色区域是空白画布，请无缝延续画面并保持其余部分不变"的说明，生成后把原图像素贴回。Qwen 会重画整幅画面，有时把原图内容缩小、平移（实测最多缩到 79%、平移 190 像素），因此贴回前先按原图区域做仿射对齐，再在每条新边内侧 64 像素做渐变融合。对不齐时返回模型自己的整幅结果，并在任务的 `warnings` 中报告 `extend_original_not_kept`。输出尺寸为原图加扩展部分；超过约 240 万像素的画布按缩小尺寸生成后放大，新区域会偏软。选路时可用 `input_mode: "extend"`（或 `"outpaint"`）。

参考图过大或为动图时不会被拒绝：默认 `input_fit` 为 `downscale`，服务会按工作流能处理的尺寸等比缩小（不修改原上传文件，缩小后的副本在任务结束时删除），动图取第一帧，并在响应 `warnings` 中以 `input_downscaled`、`input_resized` 或 `input_first_frame` 报告原始尺寸与实际尺寸。传 `input_fit: "none"` 则保持原图；工作流无法处理时以稳定的 `error.code` 拒绝（如 `input_too_large`、`input_animated`、`input_size_unsupported`、`mask_size_mismatch`），完整列表见 agent manifest 的 `error_codes`。其他支持图生图的工作流使用 `input_image` 和 `denoise` 控制参考图重绘；只给 `width` 或 `height` 时保持原图比例，两者都给且比例不同时取原图中央部分（`input_cropped` 警告），不会拉伸。

局部重绘需要原图和宽高比相同（通常同尺寸）的蒙版；尺寸不是 8 的倍数时会自动调整。原图大于工作流能绘制的尺寸时，只截取蒙版周围的区域（含部分上下文）按模型尺寸重绘，再贴回原图：输出保持原图尺寸，蒙版外的像素完全不变，并给出 `inpaint_region` 警告。默认读取蒙版红色通道：白色区域重绘、黑色区域保留；默认 `denoise` 1.0 完全重绘蒙版区域，调低则保留更多原图结构。人物参考使用 `face_image`，可以是任意一张单人图片：服务用 DWPose 找到图中最大的脸，裁成包含头发和肩膀的方图再交给 IPAdapter（本身已是头像的图片原样使用），并在 `warnings` 中以 `input_cropped` 报告裁剪范围；`input_fit: "none"` 保留整图。Anima 人物参考会编码整张（裁剪后的）参考图，`ipadapter_weight` 默认 0.6；调到 0.8–1.0 会连同服装和背景一起照搬。

参考图有六种用途：按指令修改（`input_mode` 为 `image-edit`，Qwen 编辑）、扩图（`extend`，Qwen 编辑加参数 `extend`）、按模型画风重绘（`image-to-image`，保留构图与配色但不保证人物一致）、局部重绘（`inpaint`）、人物参考（`face-reference`）和姿势（`openpose-control`，参数 `pose`）。agent manifest 的 `input_files.reference_uses` 列出每种用途对应的工作流，`route_table[].image_inputs` 列出各工作流接受的图片参数。

生成的 PNG 保留 ComfyUI 的提示词元数据（提示词、种子、模型与参数）便于复现，但其中的输入图片文件名会在任务完成前替换为 `[input image]`，不会泄露上传文件名或路径。

图片参数接受已上传的相对文件名，不接受主机绝对路径或 URL。上传图片跨重启保留，可通过输入列表查询并按文件名删除；生成结果可以重新上传，或经下面的裁剪缩放接口直接转为输入文件。

### 裁剪与缩放

`POST /project/images/transform` 在 API 进程内用 Pillow 即时处理，不加载模型、不进入任务队列（排队任务会触发模型切换并卸载当前模型），调用方不需要任何图像库：

```json
{
  "image": "/view?filename=qwen_image_2_1_00012_.png&subfolder=&type=output",
  "crop": {"aspect": "1:1", "anchor": "top"},
  "resize": {"width": 1024}
}
```

- `image`：已上传的输入文件名，或任务结果中的 `outputs[].url`。
- `crop`：像素框 `{"x", "y", "width", "height"}`；或 `{"aspect": "16:9", "anchor": "center"}` 取该比例的最大区域（anchor 可为 center、top、bottom、left、right 或 top-left 等四角）；或 `{"around": "face"}` 用 DWPose 找到图中最大的脸并取景，可选 `framing`（`head`、`head-and-shoulders`（默认）、`upper-body`）和 `aspect`（默认 1:1），找不到脸时返回 `face_not_found`。
- `resize`：只给 `width` 或 `height` 时保持比例；同时给出且比例不同时先居中裁剪再缩放，不会拉伸；也可用 `scale` 或 `max_side`（只缩小）。先裁剪后缩放。
- 结果保存为新的输入文件（`transformed/` 下），响应给出 `name`、下载 `url`、尺寸和实际执行的操作；与上传文件一样保留到被删除。放大只做插值（Lanczos），不增加细节，响应会给出 `upscaled_without_model` 警告；需要清晰放大时用下面的超分辨率接口。除 `around: "face"` 外不识别主体：`anchor: "top"` 适合保留站立人物的头部，`center` 适合场景。

### 超分辨率放大

`POST /project/images/upscale` 用放大模型把图片放大到 1–4 倍，和生成任务一样排队（轮询 `job_url` 后下载 `outputs[].url`），但不触发模型切换：每张约 3 秒，下一次生成任务从内存重新载入模型，多花几秒。

```json
{"image": "/view?filename=anima_turbo_v1_1_00007_.png&subfolder=&type=output", "scale": 2}
```

- `image` 与裁剪缩放接口相同；`scale`（大于 1，最多 4），或 `width`/`height`（同时给出且比例不同时先居中裁剪）。
- `look`：`anime`（动漫、插画、绘画，使用 4x-AnimeSharp，CC BY-NC-SA 4.0，非商用）或 `photo`（照片与写实渲染，使用 Real-ESRGAN x4plus，BSD-3）。不传时，任务结果按生成它的路由选择，上传图片用 `photo` 并给出 `upscale_look_default` 警告。
- 模型由 `scripts/download_models.ps1 -Profile upscale` 下载并校验。
- 放大模型保持画面原样，但会抹平照片的细微纹理（墙面、皮肤）。照片输出不超过约 240 万像素（如 1536×1536）时，先用裁剪缩放接口放大 2 倍，再交给 `qwen-image-2.1-edit` 并要求"保持画面不变，只恢复细节和清晰度"，纹理更自然，但耗时约 140 秒、会切换模型，细节也会改变。

### 常用接口

| 接口 | 用途 |
| --- | --- |
| `GET /.well-known/agent-manifest.json` | 发现服务能力与调用方式 |
| `GET /project/openapi.json` | 获取 OpenAPI 定义 |
| `GET /project/models`、`GET /project/models/{model_id}` | 查询模型及组件信息 |
| `GET /project/workflows`、`GET /project/workflows/{workflow_id}` | 查询工作流、参数和就绪状态 |
| `GET /project/routes`、`POST /project/resolve` | 查询与选择路由 |
| `GET /project/poses` | 搜索姿势库 |
| `POST /project/generate` | 提交命名工作流任务 |
| `GET /project/jobs/{job_id}` | 查询排队、执行状态与输出 |
| `POST /upload/image`、`GET /project/inputs` | 上传与查看输入图片 |
| `DELETE /project/inputs/{name}` | 删除指定输入图片 |
| `GET /project/resources` | 查看当前 GPU、RAM 和资源限制 |

`POST /project/prompt` 是生成接口的兼容别名。需要自行构建完整节点图的客户端，也可以使用原生 ComfyUI 的 `POST /prompt` 和 `/history`。

## 使用说明

- 所有命名工作流共用队列并支持模型热切换；切换模型时，任务状态会显示 `model_switch`。
- 队列最多接受 32 个待处理的命名任务。任务队列保存在内存中，重启服务不会保留未完成的排队任务。
- 分辨率和批量数量受各工作流的资源预算约束；实际可用显存与 RAM 可通过资源接口查看。
- CFG 为 1 时，负面提示词不参与采样。提示词风格、采样参数和可用 LoRA 以各工作流说明为准。
- 模型文件位于 `ComfyUI/models/`，输入图片位于 `ComfyUI/input/`，生成图片位于 `ComfyUI/output/`。
