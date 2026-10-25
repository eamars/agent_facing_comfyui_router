[CmdletBinding()]
param(
    [ValidateSet('qwen-image-2.1', 'moody-desire-v30', 'moody-anima-mix', 'anima-turbo', 'animagine-xl-3.1-nsfw', 'janku-illustrious', 'janku-v5-illustrious', 'one-obsession', 'deepseek-whale-girl', 'ipadapter-face-sdxl', 'controlnet-openpose-sdxl', 'moss-image-sft', 'all')]
    [string]$Profile = 'moody-desire-v30'
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Comfy = Join-Path $Root 'ComfyUI'
$ModelRoot = Join-Path $Comfy 'models'
$SigLIP2Root = Join-Path $ModelRoot 'siglip2\siglip2-base-patch16-512'
$ControlNetAuxAnnotators = Join-Path $Comfy 'custom_nodes\comfyui_controlnet_aux\ckpts\lllyasviel\Annotators'
# MOSS-Image ships as a Hugging Face component directory; its ComfyUI HF loader reads it in place.
$MossImageRoot = Join-Path $ModelRoot 'moss-image\MOSS-Image-SFT'
$MossImageRevision = 'daf706d2e1c7a6369c17b6d1658671aee6db1ebe'

$AssetSets = @{
    'qwen-image-2.1' = @(
        [pscustomobject]@{
            Name = 'Qwen-Image-2.1 Uncensored GGUF Q4_K_M diffusion model'
            Url = 'https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized/resolve/40319fb15542f0ad22921e0124a191a8a935a60a/qwen-image-2.1-UC-Q4_K_M.gguf?download=true'
            Path = (Join-Path $ModelRoot 'diffusion_models\qwen-image-2.1-UC-Q4_K_M.gguf')
            Bytes = [int64]4604558112
            Sha256 = 'E79C8A009F2ECBDB6C70FD663D9AEA9EE304A0D91F347E4169A756B8AD141B41'
        },
        [pscustomobject]@{
            Name = 'Qwen-Image-2.1 Qwen3VL 8B INT8 ConvRot text encoder'
            Url = 'https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized/resolve/40319fb15542f0ad22921e0124a191a8a935a60a/text_encoders/qwen3vl_8b_int8_convrot.safetensors?download=true'
            Path = (Join-Path $ModelRoot 'text_encoders\qwen3vl_8b_int8_convrot.safetensors')
            Bytes = [int64]9350798360
            Sha256 = '8BFD0F6E12ABF2D2D697ECC888E5E90B0D6741D6708F05799F53AFA560452E8F'
        },
        [pscustomobject]@{
            Name = 'Qwen-Image-2.1 BF16 VAE'
            Url = 'https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized/resolve/40319fb15542f0ad22921e0124a191a8a935a60a/vae/qwen_image_2.1_vae_bf16.safetensors?download=true'
            Path = (Join-Path $ModelRoot 'vae\qwen_image_2.1_vae_bf16.safetensors')
            Bytes = [int64]675509688
            Sha256 = 'BB21F7473051E1AC368515DD3F2E15CD44D7A11748EE8823E1DDCA3E4876B7C9'
        },
        [pscustomobject]@{
            Name = 'Qwen-Image-2.1 uncensored LoRA (rank 16, attention only)'
            Url = 'https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized/resolve/7959e658915f52274897fb573ffeda49de8dc16b/qwen-image-2.1-uncensored-lora.safetensors?download=true'
            Path = (Join-Path $ModelRoot 'loras\qwen-image-2.1-uncensored-lora.safetensors')
            Bytes = [int64]33586704
            Sha256 = '00EE2CB7F00A37B097640C1880ED0D17E368E30599F34BB348FEE0E8C2CC4CA1'
        }
    )
    'moss-image-sft' = @(
        # Transformer (BF16 DiT), Qwen3.5 9B text encoder, tokenizer and FLUX.2 small VAE, pinned to one revision.
        @('transformer/config.json', 240, '3291C3A1C7ADC22B8182A0DE25CA8EC97474D72715CE84ADBAB09CC0BE24F666'),
        @('transformer/diffusion_pytorch_model.safetensors.index.json', 40298, 'B609327FC75DBE5080E7E095D8B727D74D728B4F98E8B047133B0082CFA2FB03'),
        @('transformer/diffusion_pytorch_model-00001-of-00003.safetensors', 4991978016, 'E40D3444940A0B0FD7CD45BA7959A5BD3D5EFD7DDF87AAE5CE4D87CE8AA03D01'),
        @('transformer/diffusion_pytorch_model-00002-of-00003.safetensors', 4983137016, 'D106DA790EC2924FFD8AF04B4FFA8F8DE281A974B58E55649C3EC596DC8EF033'),
        @('transformer/diffusion_pytorch_model-00003-of-00003.safetensors', 189549184, '11F637AB17C7093344A6110A1E87FFDA8E56558DC55183D8E790222CFCDCF52D'),
        @('text_encoder/config.json', 2813, 'B31EE18A242BCFCCCBE740F825F6B1D3AE231B0655EEECB5DE3E98D504B7D68C'),
        @('text_encoder/model.safetensors.index.json', 64641, '0AF6FA3FA36BA8DCBB6B7AD0325A36C5B1B69862A9A1F1486B1B80385BD8F5EE'),
        @('text_encoder/model-00001-of-00004.safetensors', 4973910744, 'ABFAA3D3A8612508C85EE8F6DA9C734B862F10298B903765C4AD384177C32B0B'),
        @('text_encoder/model-00002-of-00004.safetensors', 4988364288, '9112637FA6BF4D07BAFD6AE268852B18D3D0C1242F4EE901C5579006E5BD4894'),
        @('text_encoder/model-00003-of-00004.safetensors', 4954202112, '956F1EF51D0CFE610B0D22987C0B7821C36E3CE228ABCECCEF523518BCF079DC'),
        @('text_encoder/model-00004-of-00004.safetensors', 1869001848, '9E21D20E8ADDE88C3E0B65D4A0D251F143D98EE9D52DB36B4348361091E4E208'),
        @('tokenizer/tokenizer.json', 19989325, '06B9509352D2AF50381AB2247E083B80D32D5C0ABA91C272CA9FF729B6A0E523'),
        @('tokenizer/tokenizer_config.json', 1123, 'BEE8EBA30F0EB4AF73C0FE2CD06D0F89B657D7819941C438157EC42F7C80EA87'),
        @('tokenizer/chat_template.jinja', 7756, 'A4AEE8AFCF2E0711942CF848899BE66016F8D14A889FF9EDE07BCA099C28F715'),
        @('vae/config.json', 838, 'CC77F3F0CE0357039C71018DC19EDAA4FCCEB0C2F7488FDC2B9C0580D6BAAFA1'),
        @('vae/diffusion_pytorch_model.safetensors', 249521372, 'C3BDF94DC67EC69B30EA9FE1FA8340085679CAE64D019F7C7E4A9921F1A7F17A')
    ) | ForEach-Object {
        [pscustomobject]@{
            Name = "MOSS-Image-SFT $($_[0])"
            Url = "https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/resolve/$MossImageRevision/$($_[0])?download=true"
            Path = (Join-Path $MossImageRoot $_[0].Replace('/', '\'))
            Bytes = [int64]$_[1]
            Sha256 = $_[2]
        }
    }
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
    'janku-illustrious' = @(
        [pscustomobject]@{
            Name = 'JANKU v7.77 canonical Illustrious/SDXL checkpoint (janxd, Civitai model 1277670, version 2786084)'
            Url = 'https://huggingface.co/LillyCherry/JANKUTrainedChenkinNoobai_v777/resolve/0226c4f27bfbcfdb744224073ad369288637974c/JANKUTrainedChenkinNoobai_v777.safetensors?download=true'
            Path = (Join-Path $ModelRoot 'checkpoints\JANKU_v7_77.safetensors')
            Bytes = [int64]6938040674
            Sha256 = '88177D224CE97F60CF3E908F87902F913ED4562BD79D237F84238A4354601EFB'
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
    'one-obsession' = @(
        [pscustomobject]@{
            Name = 'One Obsession v24 canonical Illustrious/SDXL checkpoint (maxfeifei8, Civitai model 1318945, version 3218603)'
            Url = 'https://huggingface.co/RunningHubAI/rh-one-obsession-v24-checkpoint/resolve/372518ef0b955be4cd6de0907e7a1dd9d928c262/oneObsession_v24.safetensors?download=true'
            Path = (Join-Path $ModelRoot 'checkpoints\OneObsession_v24.safetensors')
            Bytes = [int64]6938040682
            Sha256 = '2FAE33D7F0D23920D145897827E6015AC098A2B90CB682D07846AE762213100B'
        }
    )
    'deepseek-whale-girl' = @(
        # Civitai-only files with no Hugging Face mirror; set CIVITAI_TOKEN before downloading.
        [pscustomobject]@{
            Name = 'DeepSeek whale girl JANKU LoRA (yrx0110121, Civitai model 2595806, version 2915962)'
            Url = 'https://civitai.com/api/download/models/2915962?fileId=2794189'
            Path = (Join-Path $ModelRoot 'loras\DeepSeek_WhaleGirl_ChenkinNoobXL.safetensors')
            Bytes = [int64]228452020
            Sha256 = '54A721895A0AA6A38F1A48C232916A091A9C881941D024779E3C12A061344AEF'
        },
        [pscustomobject]@{
            Name = 'DeepSeek whale girl ZipZipPipe-style Anima 2B LoRA (HermitST, Civitai model 2919135, version 3302634)'
            Url = 'https://civitai.com/api/download/models/3302634?fileId=3187626'
            Path = (Join-Path $ModelRoot 'loras\DeepSeek_WhaleGirl_ZipZipPipe_Anima2B.safetensors')
            Bytes = [int64]183628056
            Sha256 = 'B7CEDCC806C20A788DA2A8E1DD06EBAE1A1F6B2FF710FE92FB5706648779C45D'
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
        $auth = if ($Asset.Url -like 'https://civitai.com/*' -and $env:CIVITAI_TOKEN) { @('-H', "Authorization: Bearer $env:CIVITAI_TOKEN") } else { @() }
        & curl.exe -L --fail --retry 10 --retry-delay 5 --continue-at - -A 'Mozilla/5.0' @auth -o $Asset.Path $Asset.Url
        if ($LASTEXITCODE -ne 0) { throw "Download failed for $($Asset.Name) with exit code $LASTEXITCODE" }
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Asset.Path).Hash.ToUpperInvariant()
    $size = (Get-Item -LiteralPath $Asset.Path).Length
    if ($size -ne $Asset.Bytes -or $actual -ne $Asset.Sha256) {
        throw "Verification failed for $($Asset.Path): size=$size hash=$actual expected_size=$($Asset.Bytes) expected_hash=$($Asset.Sha256)"
    }
    Write-Host ("[verified] {0}  {1:N2} GB  SHA256 {2}" -f $Asset.Path, ($size / 1GB), $actual)
}

$profiles = if ($Profile -eq 'all') { @('moody-desire-v30', 'moody-anima-mix', 'anima-turbo', 'animagine-xl-3.1-nsfw', 'janku-illustrious', 'janku-v5-illustrious', 'one-obsession', 'deepseek-whale-girl', 'ipadapter-face-sdxl', 'controlnet-openpose-sdxl', 'qwen-image-2.1', 'moss-image-sft') } else { @($Profile) }
foreach ($profile in $profiles) {
    Write-Host "=== model profile: $profile ==="
    foreach ($asset in $AssetSets[$profile]) { Download-Verified $asset }
}

Write-Host ''
Write-Host 'Selected model profile assets are present and verified.'
