# 打包交接物：原始数据 / results / 说明文件
# 用法（项目根目录）:
#   powershell -File scripts\package_handoff.ps1 -IncludeData
#   powershell -File scripts\package_handoff.ps1 -IncludeData -IncludeResults
#   powershell -File scripts\package_handoff.ps1 -DataRoot "D:\path\to\dataAnalysis"
param(
    [switch]$IncludeData,
    [switch]$IncludeResults,
    [string]$DataRoot = "",
    [string]$OutDir = "handoff"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$OutRoot = Join-Path $ProjectRoot $OutDir
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

function Read-DataRoot {
    $pathsFile = Join-Path $ProjectRoot "configs\paths.json"
    if (-not (Test-Path $pathsFile)) { return "" }
    $cfg = Get-Content $pathsFile -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($cfg.PSObject.Properties.Name -contains "data_root") { return [string]$cfg.data_root }
    return ""
}

if (-not $DataRoot) { $DataRoot = Read-DataRoot }

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

if ($IncludeData) {
    if (-not $DataRoot) {
        Write-Error "未找到 data_root。请传 -DataRoot 或先配置 configs/paths.json"
    }
    if (-not (Test-Path $DataRoot)) {
        Write-Error "data_root 不存在: $DataRoot"
    }
    $zipData = Join-Path $OutRoot ("seismo-fluid-data_{0}.zip" -f $stamp)
    $items = @("OriData", "OriData2", "OriData3") | ForEach-Object { Join-Path $DataRoot $_ } | Where-Object { Test-Path $_ }
    if (-not $items) {
        Write-Error "在 $DataRoot 下未找到 OriData / OriData2 / OriData3"
    }
    Write-Host "打包数据: $($items -join '; ')"
    Compress-Archive -Path $items -DestinationPath $zipData -Force
    Write-Host "已生成 $zipData"
}

if ($IncludeResults) {
    $resDir = Join-Path $ProjectRoot "results"
    if (-not (Test-Path $resDir)) {
        Write-Error "results/ 不存在"
    }
    $zipRes = Join-Path $OutRoot ("seismo-fluid-results_{0}.zip" -f $stamp)
    # 排除 README 可选；整体打包 results 树
    Compress-Archive -Path (Join-Path $resDir "*") -DestinationPath $zipRes -Force
    Write-Host "已生成 $zipRes"
}

# 始终写出简短发送清单
$readme = Join-Path $OutRoot "SEND_CHECKLIST.md"
@"
# 发送清单（$stamp）

## 仓库
- 确认已 push 到远程后把 URL 给对方
- 若本机有未提交 v6.1 脚本/笔记，需先 commit+push 或改用工作区 zip

## 可选附件
$(if ($IncludeData) { "- seismo-fluid-data_*.zip （原始数据）`n" } else { "" })$(if ($IncludeResults) { "- seismo-fluid-results_*.zip （实验结果）`n" } else { "" })
## 对方必读
- README.md
- docs/交接说明.md
- AGENTS.md

## 验收
1. python -m pip install -r requirements.txt
2. python smoke_test.py
3. 配置 configs/paths.json（复制 paths.example.json）
4. python web/app.py → http://127.0.0.1:8002/earthquake
"@ | Set-Content -Path $readme -Encoding UTF8

Write-Host "清单已写: $readme"
Write-Host "完成。目录: $OutRoot"
