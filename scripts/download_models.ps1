[CmdletBinding()]
param(
    [ValidateSet('qwen-image-2.1', 'moody-desire-v30', 'moody-anima-mix', 'anima-turbo', 'animagine-xl-3.1-nsfw', 'blue-archive-style-sdxl', 'blue-archive-character-illustrious-xl', 'janku-v5-illustrious', 'one-obsession-v20-bold', 'ipadapter-face-sdxl', 'controlnet-openpose-sdxl', 'all')]
    [string]$Profile = 'moody-desire-v30'
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Comfy = Join-Path $Root 'ComfyUI'
$ModelRoot = Join-Path $Comfy 'models'
$SigLIP2Root = Join-Path $ModelRoot 'siglip2\siglip2-base-patch16-512'
$ControlNetAuxAnnotators = Join-Path $Comfy 'custom_nodes\comfyui_controlnet_aux\ckpts\lllyasviel\Annotators'

$AssetSets = @{
    'qwen-image-2.1' = @(
        [pscustomobject]@{
            Name = 'Qwen-Image-2.1 Uncensored GGUF Q4_K_M diffusion model'
            Url = 'https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/resolve/40319fb15542f0ad22921e0124a191a8a935a60a/qwen-image-2.1-UC-Q4_K_M.gguf?download=true'
            Path = (Join-Path $ModelRoot 'diffusion_models\qwen-image-2.1-UC-Q4_K_M.gguf')
            Bytes = [int64]4604558112
            Sha256 = 'E79C8A009F2ECBDB6C70FD663D9AEA9EE304A0D91F347E4169A756B8AD141B41'
        },
        [pscustomobject]@{
            Name = 'Qwen-Image-2.1 Qwen3VL 8B INT8 ConvRot text encoder'
            Url = 'https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/resolve/40319fb15542f0ad22921e0124a191a8a935a60a/text_encoders/qwen3vl_8b_int8_convrot.safetensors?download=true'
            Path = (Join-Path $ModelRoot 'text_encoders\qwen3vl_8b_int8_convrot.safetensors')
            Bytes = [int64]9350798360
            Sha256 = '8BFD0F6E12ABF2D2D697ECC888E5E90B0D6741D6708F05799F53AFA560452E8F'
        },
        [pscustomobject]@{
            Name = 'Qwen-Image-2.1 BF16 VAE'
            Url = 'https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/resolve/40319fb15542f0ad22921e0124a191a8a935a60a/vae/qwen_image_2.1_vae_bf16.safetensors?download=true'
            Path = (Join-Path $ModelRoot 'vae\qwen_image_2.1_vae_bf16.safetensors')
            Bytes = [int64]675509688
            Sha256 = 'BB21F7473051E1AC368515DD3F2E15CD44D7A11748EE8823E1DDCA3E4876B7C9'
        }
    )
    'moody-anima-mix' = @(
        [pscustomobject]@{
            Name = 'Moody Anima Mix v1.0 checkpoint (Civitai model 2700077, file 2911005)'
            Url = 'https://huggingface.co/minyan243/lotc/resolve/main/moodyAnimaMix_v10.safetensors'
            Path = (Join-Path $ModelRoot 'diffusion_models\moodyAnimaMix_v10.safetensors')
            Bytes = [int64]4182230672
            Sha256 = '5368C9B1318C783D7215492002CD1BCA94895D1DCA8C780EF24F75536FD341F8'
        },
        [pscustomobject]@{
            Name = 'Official Anima Qwen 3 0.6B text encoder'
            Url = 'https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors'
            Path = (Join-Path $ModelRoot 'text_encoders\qwen_3_06b_base.safetensors')
            Bytes = [int64]1192135096
            Sha256 = 'CD2A512003E2F9F3CD3C32A9C3573F820BB28C940F73C57B1DDAA983D9223EBA'
        },
        [pscustomobject]@{
            Name = 'Official Anima Qwen image VAE'
            Url = 'https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors'
            Path = (Join-Path $ModelRoot 'vae\qwen_image_vae.safetensors')
            Bytes = [int64]253806246
            Sha256 = 'A70580F0213E67967EE9C95F05BB400E8FB08307E017A924BF3441223E023D1F'
        }
    )
    'moody-desire-v30' = @(
        [pscustomobject]@{
            Name = 'Moody Desire Mix v3.0 full checkpoint (Civitai file 2943512)'
            Url = 'https://huggingface.co/catlover1937/moody-desire-mix/resolve/main/moody-desire-v3.1_00001_.safetensors'
            Path = (Join-Path $ModelRoot 'diffusion_models\moodyDesireMix_v30_2943512.safetensors')
            Bytes = [int64]18157246296
            Sha256 = '76FAEF1511441EEA2049A593603AA883F59B1FEB2EB8A296D737771AD4433CEE'
        },
        [pscustomobject]@{
            Name = 'Flux.2 Klein Qwen3 8B FP8 mixed text encoder'
            Url = 'https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors'
            Path = (Join-Path $ModelRoot 'text_encoders\qwen_3_8b_fp8mixed.safetensors')
            Bytes = [int64]8664848742
            Sha256 = 'ABAD16806E0CBABC54E0325D6565847443FE396D5F0BE38BB3CD3FE75A1201D6'
        },
        [pscustomobject]@{
            Name = 'Flux.2 Klein Qwen3 8B full-precision text encoder (optional high-precision route asset)'
            Url = 'https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/text_encoders/qwen_3_8b.safetensors'
            Path = (Join-Path $ModelRoot 'text_encoders\qwen_3_8b.safetensors')
            Bytes = [int64]16381517176
            Sha256 = 'F0FF9239D56269CA1D05E5F86DA6A79FAC111AF464955681F11C7AB0EC5EF6C1'
        },
        [pscustomobject]@{
            Name = 'Flux.2 Klein VAE'
            Url = 'https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/vae/flux2-vae.safetensors'
            Path = (Join-Path $ModelRoot 'vae\flux2-vae.safetensors')
            Bytes = [int64]336211292
            Sha256 = '868FE7B343CC8F3A19DBCFCafBC3D5F888802BE3F89BD81B65B3621A066CE8F3'.ToUpperInvariant()
        }
    )
    'anima-turbo' = @(
        [pscustomobject]@{
            Name = 'Official Anima Turbo v1.1 diffusion model (BF16, highest published Turbo precision)'
            Url = 'https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-turbo-v1.1.safetensors'
            Path = (Join-Path $ModelRoot 'diffusion_models\anima-turbo-v1.1.safetensors')
            Bytes = [int64]4182230656
            Sha256 = 'FBA11953276B57EDF59D1DC4F1857AC05AA079C56F982B4D7C20298D57D3F7EB'
        },
        [pscustomobject]@{
            Name = 'Anima IP-Adapter Character Reference v10 weights'
            Url = 'https://huggingface.co/LuciferTC/Anima-IP-Adapter/resolve/main/ip_adapter-Character_Reference-10.safetensors'
            Path = (Join-Path $ModelRoot 'ipadapter\anima-ip-adapter-character-reference-10.safetensors')
            Bytes = [int64]503229576
            Sha256 = 'DC9612CC28B55F00BA39F6169BFE4871130BE028A55682AB54DD51360ACB8232'
        },
        [pscustomobject]@{
            Name = 'SigLIP2 base patch16 512 vision encoder'
            Url = 'https://huggingface.co/google/siglip2-base-patch16-512/resolve/main/model.safetensors'
            Path = (Join-Path $SigLIP2Root 'model.safetensors')
            Bytes = [int64]1503344520
            Sha256 = 'FE0E601C625E69EED8E73500D39E9B6164403FE03DB8048E87913C3CEFBBB3FE'
        },
        [pscustomobject]@{
            Name = 'SigLIP2 vision encoder config'
            Url = 'https://huggingface.co/google/siglip2-base-patch16-512/resolve/main/config.json'
            Path = (Join-Path $SigLIP2Root 'config.json')
            Bytes = [int64]276
            Sha256 = 'A15BC39FBCD92498DEA6D5D3CB2B0D8B9BA389D6A3F2B565EA66128A2D141934'
        },
        [pscustomobject]@{
            Name = 'SigLIP2 image processor config'
            Url = 'https://huggingface.co/google/siglip2-base-patch16-512/resolve/main/preprocessor_config.json'
            Path = (Join-Path $SigLIP2Root 'preprocessor_config.json')
            Bytes = [int64]394
            Sha256 = '63FF380D3E424F93E6FBCA5CC8E74EEED882E96FD6BE2B7728AED308F3AD1513'
        }
    )
    'animagine-xl-3.1-nsfw' = @(
        [pscustomobject]@{
            Name = 'Animagine XL 3.1 SDXL checkpoint'
            Url = 'https://huggingface.co/cagliostrolab/animagine-xl-3.1/resolve/main/animagine-xl-3.1.safetensors?download=true'
            Path = (Join-Path $ModelRoot 'checkpoints\Animagine_XL_3.1.safetensors')
            Bytes = [int64]6938325776
            Sha256 = 'E3C47AEDB06418C6C331443CD89F2B3B3B34B7ED2102A3D4C4408A8D35AAD6B0'
        },
        [pscustomobject]@{
            Name = 'NIKKE Goddess of Victory profile Animagine/SDXL LoRA'
            Url = 'https://huggingface.co/Muapi/nikke-goddess-of-victory-profile-concept-lora-animagine/resolve/main/nikke-goddess-of-victory-profile-concept-lora-animagine.safetensors?download=true'
            Path = (Join-Path $ModelRoot 'loras\NIKKES_GoddessOfVictory_Concept.safetensors')
            Bytes = [int64]57430604
            Sha256 = 'B1873E7F09FEBFB8DBF4F8931F6CF62BDA7C74D719BB96797672A9F9792EE50F'
        },
        [pscustomobject]@{
            Name = 'Rapi NIKKE Pony-XL/SDXL-shaped LoRA (Civitai version 427079)'
            Url = 'https://civitai.com/api/download/models/427079?fileId=348255'
            Path = (Join-Path $ModelRoot 'loras\NIKKES_Rapi.safetensors')
            Bytes = [int64]114474060
            Sha256 = 'FE9F99F0C9908673C3F5B227C91BCCD7744FC3FCC95A4ED1E3842F44A82B55A0'
        }
    )
    'blue-archive-style-sdxl' = @(
        [pscustomobject]@{
            Name = 'Blue Archive Game CG Style LoCon (Animagine 3.1)'
            Url = 'https://huggingface.co/licyk/sd-lora/resolve/main/sdxl/style/ag31_style_ba_v1-000016.safetensors?download=true'
            Path = (Join-Path $ModelRoot 'loras\BlueArchive_GameCG_Style_Animagine31.safetensors')
            Bytes = [int64]21995392
            Sha256 = 'D2E9468E1FE6B2134F58C883C984E9845FA197E65EF5B2CF54716C251DCDF589'
        }
    )
    'blue-archive-character-illustrious-xl' = @(
        [pscustomobject]@{
            Name = 'Illustrious XL v0.1 checkpoint for the Asuna character route'
            Url = 'https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/main/Illustrious-XL-v0.1.safetensors?download=true'
            Path = (Join-Path $ModelRoot 'checkpoints\Illustrious_XL_v0.1.safetensors')
            Bytes = [int64]6938040760
            Sha256 = '3E15BA00387DB678AB4A099F75771C4F5AC67FDA9E7100A01D263EAF30145AA9'
        },
        [pscustomobject]@{
            Name = 'Asuna Ichinose Blue Archive 4-outfit IllustriousXL LoRA'
            Url = 'https://huggingface.co/Muapi/asuna-ichinose-blue-archive-4-outfits-illustriousxl/resolve/main/asuna-ichinose-blue-archive-4-outfits-illustriousxl.safetensors?download=true'
            Path = (Join-Path $ModelRoot 'loras\Asuna_Ichinose_IllustriousXL.safetensors')
            Bytes = [int64]76036486
            Sha256 = '24E04396471196283291B664FFB3DE9E59B770F4CBF99FF6573E6816F8B69685'
        }
    )
    'janku-v5-illustrious' = @(
        [pscustomobject]@{
            Name = 'JANKU v5.0 canonical Illustrious/SDXL checkpoint (janxd, Civitai model 1277670, version 1896532)'
            Url = 'https://huggingface.co/Quiho/best-from-civitai/resolve/main/JANKUTrainedNoobaiRouwei_v50.safetensors?download=true'
            Path = (Join-Path $ModelRoot 'checkpoints\JANKU_v5_0.safetensors')
            Bytes = [int64]6938040674
            Sha256 = '217DAAE812DDC3E501C1E972F88E01D5F3BAE8A4C3EAD2F087E8A9CD4B4CDED5'
        }
    )
    'one-obsession-v20-bold' = @(
        [pscustomobject]@{
            Name = 'One Obsession v20_BOLD canonical Illustrious/SDXL checkpoint (maxfeifei8, Civitai model 1318945)'
            Url = 'https://huggingface.co/Quiho/best-from-civitai/resolve/main/oneObsession_v20Bold.safetensors?download=true'
            Path = (Join-Path $ModelRoot 'checkpoints\OneObsession_v20Bold.safetensors')
            Bytes = [int64]6938040682
            Sha256 = '7CE890BD4225CAE66B73F5291D47DFC07C60012630DBA21CD0536D041254CD8B'
        }
    )
    'ipadapter-face-sdxl' = @(
        [pscustomobject]@{
            Name = 'IPAdapter Plus Face SDXL ViT-H adapter'
            Url = 'https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors?download=true'
            Path = (Join-Path $ModelRoot 'ipadapter\ip-adapter-plus-face_sdxl_vit-h.safetensors')
            Bytes = [int64]847517512
            Sha256 = '677AD8860204F7D0BFBA12D29E6C31DED9BEEFDF3E4BBD102518357D31A292C1'
        },
        [pscustomobject]@{
            Name = 'IPAdapter SDXL face CLIP ViT-H image encoder'
            Url = 'https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors?download=true'
            Path = (Join-Path $ModelRoot 'clip_vision\CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors')
            Bytes = [int64]2528373448
            Sha256 = '6CA9667DA1CA9E0B0F75E46BB030F7E011F44F86CBFB8D5A36590FCD7507B030'
        }
    )
    'controlnet-openpose-sdxl' = @(
        [pscustomobject]@{
            Name = 'Xinsir ControlNet OpenPose SDXL 1.0'
            Url = 'https://huggingface.co/xinsir/controlnet-openpose-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors?download=true'
            Path = (Join-Path $ModelRoot 'controlnet\xinsir-controlnet-openpose-sdxl-1.0.safetensors')
            Bytes = [int64]2502139104
            Sha256 = 'B8524E557A7DF60D081F5D4A0EB109967D107DF217943BF88C2D99B9EBCC06C5'
        },
        [pscustomobject]@{
            Name = 'OpenPose body detector annotator weight'
            Url = 'https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth?download=true'
            Path = (Join-Path $ControlNetAuxAnnotators 'body_pose_model.pth')
            Bytes = [int64]209267595
            Sha256 = '25A948C16078B0F08E236BDA51A385D855EF4C153598947C28C0D47ED94BB746'
        },
        [pscustomobject]@{
            Name = 'OpenPose hand detector annotator weight'
            Url = 'https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth?download=true'
            Path = (Join-Path $ControlNetAuxAnnotators 'hand_pose_model.pth')
            Bytes = [int64]147341049
            Sha256 = 'B76B00D1750901ABD07B9F9D8C98CC3385B8FE834A26D4B4F0AAD439E75FC600'
        },
        [pscustomobject]@{
            Name = 'OpenPose face detector annotator weight'
            Url = 'https://huggingface.co/lllyasviel/Annotators/resolve/main/facenet.pth?download=true'
            Path = (Join-Path $ControlNetAuxAnnotators 'facenet.pth')
            Bytes = [int64]153718792
            Sha256 = '8BEB52E548624FFCC4AED12AF7AEE7DCBFAEEA420C75609FEE999FE7ADD79D43'
        }
    )
}

function Download-Verified($Asset) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Asset.Path) | Out-Null
    $existing = if (Test-Path -LiteralPath $Asset.Path) { Get-Item -LiteralPath $Asset.Path } else { $null }
    if ($existing -and $existing.Length -eq $Asset.Bytes) {
        Write-Host "[verify] $($Asset.Name) already has expected size; hashing..."
    } else {
        $resume = if ($existing) { 'resume' } else { 'new' }
        Write-Host "[download/$resume] $($Asset.Name)"
        & curl.exe -L --fail --retry 10 --retry-delay 5 --continue-at - -A 'Mozilla/5.0' -o $Asset.Path $Asset.Url
        if ($LASTEXITCODE -ne 0) { throw "Download failed for $($Asset.Name) with exit code $LASTEXITCODE" }
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Asset.Path).Hash.ToUpperInvariant()
    $size = (Get-Item -LiteralPath $Asset.Path).Length
    if ($size -ne $Asset.Bytes -or $actual -ne $Asset.Sha256) {
        throw "Verification failed for $($Asset.Path): size=$size hash=$actual expected_size=$($Asset.Bytes) expected_hash=$($Asset.Sha256)"
    }
    Write-Host ("[verified] {0}  {1:N2} GB  SHA256 {2}" -f $Asset.Path, ($size / 1GB), $actual)
}

$profiles = if ($Profile -eq 'all') { @('moody-desire-v30', 'moody-anima-mix', 'anima-turbo', 'animagine-xl-3.1-nsfw', 'blue-archive-style-sdxl', 'blue-archive-character-illustrious-xl', 'janku-v5-illustrious', 'one-obsession-v20-bold', 'ipadapter-face-sdxl', 'controlnet-openpose-sdxl', 'qwen-image-2.1') } else { @($Profile) }
foreach ($profile in $profiles) {
    Write-Host "=== model profile: $profile ==="
    foreach ($asset in $AssetSets[$profile]) { Download-Verified $asset }
}

Write-Host ''
Write-Host 'Selected model profile assets are present and verified.'
