# ComfyUI 本地图像生成服务

一个面向个人使用和 AI Agent 的本地图像生成服务。通过 ComfyUI 网页或统一 HTTP API，使用不同模型完成文生图、图生图、指令编辑、局部重绘、人物参考和姿态控制。

项目提供可查询的模型与工作流目录。调用方可以指定工作流，也可以按题材、风格和输入类型选择路由，无需自行拼装 ComfyUI 节点。模型在任务之间自动热切换，不需要重启服务。

## 能做什么

- **图像生成**：使用 Qwen-Image、Anima、Flux.2/Klein 和 SDXL/Illustrious 系列模型生成写实图像、动漫人物和插画。
- **参考图编辑**：通过 Qwen-Image-2.1 输入一张图片和自然语言指令，修改颜色、服装、场景等内容。
- **图生图与局部重绘**：使用参考图生成变体，或提供蒙版重绘指定区域并保留蒙版外的原图像素。
- **人物与姿态参考**：通过 IPAdapter 参考人物外观，通过 OpenPose ControlNet 控制身体姿态。
- **自动选路**：根据题材、内容模式、风格和输入方式，从可用工作流中选择合适的模型组合。
- **队列与热切换**：串行处理生成任务，同一工作流复用模型缓存，切换工作流时自动释放并加载所需模型。
- **Agent 接入**：提供能力发现、参数说明、图片上传、任务状态和结果下载接口，也保留原生 ComfyUI 接口。

当前部署面向 Windows、Python 3.12 和 RTX 4090。启动器锁定本机指定的 4090，支持 RAM offload 和 CPU VAE；模型权重与生成过程均在本机运行。

## 支持的模型

下表列出可通过命名工作流调用的模型组合。图生图指基于参考图的重采样；Qwen 的指令编辑使用独立工作流。

| 模型 / 组合 | 主要用途 | 支持方式 | 工作流 ID |
| --- | --- | --- | --- |
| Qwen-Image-2.1 · Uncensored GGUF Q4_K_M | 自然语言生成、图中文字、参考图指令编辑 | 文生图、单图编辑，SFW / NSFW 路由 | `qwen-image-2.1-t2i`、`qwen-image-2.1-edit` |
| Anima Turbo v1.1 · 官方 BF16 | 通用动漫、人物与插画 | 文生图、图生图 | `anima-turbo-v1-1` |
| Moody Anima Mix v1.0 | 低步数动漫生成 | 文生图、图生图 | `moody-anima-mix` |
| Moody Desire Mix v3.0 · Flux.2/Klein | 动漫、人物肖像 | 文生图、图生图 | `moody-desire-v30` |
| Animagine XL 3.1 + NIKKE LoRA | NIKKE 风格；可选 Rapi 角色适配 | 文生图、图生图 | `animagine-xl-3.1-nsfw` |
| Animagine XL 3.1 + Blue Archive 风格 LoRA | 蔚蓝档案游戏 CG 风格 | 文生图、图生图 | `blue-archive-style-sdxl` |
| Illustrious XL v0.1 + Asuna LoRA | 一之濑明日奈及其服装表现 | 文生图、图生图 | `blue-archive-character-illustrious-xl` |
| JANKU v5.0 · Illustrious/NoobAI 系 | 色彩鲜明、光泽感较强的动漫插画 | 文生图、图生图 | `janku-v5-illustrious` |
| One Obsession v20_BOLD · Illustrious 系 | 纹理、戏剧性光影与轻度 2.5D 风格 | 文生图、图生图 | `one-obsession-v20-bold` |

动漫路由和 Qwen-Image-2.1 Uncensored 路由均声明支持 SFW / NSFW 内容模式。内容模式用于路由选择，具体画面由调用方的提示词决定。

### 局部重绘、人物参考与姿态控制

| 工作流 ID | 模型 / 组件 | 用途 | 除提示词外的必需输入 |
| --- | --- | --- | --- |
| `anima-turbo-inpaint` | Anima Turbo v1.1 | 蒙版区域重绘 | `input_image`、`mask_image` |
| `moody-anima-inpaint` | Moody Anima Mix | 蒙版区域重绘 | `input_image`、`mask_image` |
| `anima-turbo-face-reference-ipadapter` | Anima Turbo + Character Reference IPAdapter | 人物外观参考 | `face_image` |
| `sdxl-face-reference-ipadapter` | Animagine XL 3.1 + IPAdapter Plus Face | 面部外观参考 | `face_image` |
| `sdxl-openpose-controlnet` | Animagine XL 3.1 + Xinsir OpenPose | 身体姿态参考 | `control_image` |

人物参考用于引导外观，不保证身份或五官完全一致。姿态控制主要约束身体姿势，人物、服装和背景仍由提示词描述。

### 如何选择

通用 SFW 动漫请求优先使用 Anima Turbo，通用 NSFW 动漫请求优先使用 JANKU。明确指定 NIKKE、Blue Archive 或模型风格时，路由器优先匹配对应组合。Asuna 角色工作流需要显式指定工作流 ID。

使用 Qwen 时，可指定 `domain: "qwen"`，并选择 `text-to-image` 或 `image-to-image`。局部重绘、人物参考和姿态控制分别使用 `inpaint`、`face-reference`、`openpose-control`。

全部工作流、默认参数和输入要求见 [工作流目录](workflows/catalog.json)；模型组件、来源与文件信息见 [模型表](workflows/models.json)。运行时可查询 `GET /project/workflows` 获取实际可用状态。

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
| Qwen-Image-2.1 Uncensored GGUF Q4_K_M 生成与编辑 | `qwen-image-2.1` |
| Anima Turbo、局部重绘与 Anima 人物参考 | `anima-turbo` |
| Moody Anima Mix 与局部重绘 | `moody-anima-mix` |
| Moody Desire Mix | `moody-desire-v30` |
| Animagine XL 3.1 / NIKKE | `animagine-xl-3.1-nsfw` |
| Blue Archive 风格 | `blue-archive-style-sdxl` |
| Asuna 角色 | `blue-archive-character-illustrious-xl` |
| JANKU v5.0 | `janku-v5-illustrious` |
| One Obsession v20_BOLD | `one-obsession-v20-bold` |
| SDXL 人物参考附加组件 | `ipadapter-face-sdxl` |
| SDXL 姿态控制附加组件 | `controlnet-openpose-sdxl` |

SDXL 人物参考与姿态控制还需要 Animagine XL 3.1 基础模型，可通过 `animagine-xl-3.1-nsfw` Profile 安装。

Qwen-Image-2.1 使用 abenzerps 发布的 Uncensored Q4_K_M GGUF 扩散模型，并配套 Qwen3VL INT8 文本编码器与 BF16 VAE。`setup_comfyui.ps1` 会安装固定版本的 ComfyUI-GGUF 节点；目标模型及配套文件由 Qwen profile 下载并校验。原官方 INT8 扩散文件保留在本地作回滚，不再被命名工作流加载。部署细节见 [Qwen 2.1 替换记录](docs/qwen-image-2.1-uncensored-migration.md)。

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

Qwen 文生图默认使用 1024×1024、25 步、Euler/simple、CFG 1。各工作流均提供自己的默认参数，可通过工作流详情中的 `parameter_schema` 查询允许调整的字段。省略 `seed` 时会随机生成，并在响应中返回实际值。

如果希望自动选路，先调用 `POST /project/resolve`：

```json
{
  "domain": "anime",
  "content_mode": "sfw",
  "input_type": "text",
  "input_mode": "text-to-image"
}
```

响应中的 `selected_workflow` 可用于生成请求。选路只返回工作流及其提示词建议，不会启动生成；角色、服装、动作和场景等细节写入 `parameters.prompt`。

生成响应提供 `job_id` 和 `job_url`。轮询 `job_url`，直到状态为 `succeeded` 或 `failed`；成功时通过 `outputs[].url` 获取图片。请求顶层添加 `"dry_run": true` 可检查参数和依赖而不生成图像，响应会说明工作流是否就绪及图是否通过验证。

请求超过工作流的 `max_total_pixels` 时不会被拒绝，响应 `warnings` 中会出现 `over_pixel_budget`，任务照常运行。若任务因内存或显存不足失败，`error.code` 为 `out_of_memory`。请把错误和 `GET /project/resources` 的当前空闲内存告诉请求者，由其释放内存或缩小尺寸后再重新提交。

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

Qwen 编辑接口支持单张参考图，输出尺寸随原图按 32 像素取整，不接受宽高或批量数量覆盖。其他支持图生图的工作流使用 `input_image` 和 `denoise` 控制参考图重绘。

局部重绘需要原图和同尺寸蒙版，宽高均为 8 的倍数。默认读取蒙版红色通道：白色区域重绘、黑色区域保留。人物参考使用 `face_image`，姿态参考使用 `control_image`。

图片参数接受已上传的相对文件名，不接受主机绝对路径或 URL。上传图片跨重启保留，可通过输入列表查询并按文件名删除；生成结果如需作为下一次输入，需要重新上传。

### 常用接口

| 接口 | 用途 |
| --- | --- |
| `GET /.well-known/agent-manifest.json` | 发现服务能力与调用方式 |
| `GET /project/openapi.json` | 获取 OpenAPI 定义 |
| `GET /project/models`、`GET /project/models/{model_id}` | 查询模型及组件信息 |
| `GET /project/workflows`、`GET /project/workflows/{workflow_id}` | 查询工作流、参数和就绪状态 |
| `GET /project/routes`、`POST /project/resolve` | 查询与选择路由 |
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
