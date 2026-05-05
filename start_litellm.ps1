# Start the LiteLLM proxy for the multi-agent AutoML pipeline.
# Run this in a separate terminal before starting the pipeline.
#
# Usage:  .\start_litellm.ps1

$litellm = "C:\Users\DATANEST\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\Scripts\litellm.exe"
$config  = Join-Path $PSScriptRoot "config.yaml"
$envFile = Join-Path $PSScriptRoot ".env"

# Load .env into the current process so LiteLLM inherits OPENAI_API_KEY etc.
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        # Skip blank lines and comments
        if ($_ -match '^\s*([^#\s][^=]*)=(.*)$') {
            $key   = $matches[1].Trim()
            $value = $matches[2].Trim()
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
    Write-Host ".env loaded — environment variables set"
} else {
    Write-Warning ".env not found. OPENAI_API_KEY may not be set. Copy .env.example to .env and fill in your key."
}

if (-not (Test-Path $litellm)) {
    Write-Error "litellm.exe not found at: $litellm"
    Write-Host "Install it with:  pip install litellm"
    exit 1
}

if (-not (Test-Path $config)) {
    Write-Error "config.yaml not found at: $config"
    exit 1
}

# Verify the key was loaded
if (-not $env:OPENAI_API_KEY) {
    Write-Warning "OPENAI_API_KEY is still empty. cloud-model calls will fail with 401."
} else {
    Write-Host "OPENAI_API_KEY loaded (starts with: $($env:OPENAI_API_KEY.Substring(0, [Math]::Min(8, $env:OPENAI_API_KEY.Length)))...)"
}

Write-Host "`nStarting LiteLLM proxy on http://localhost:4000 ..."
Write-Host "Press Ctrl+C to stop.`n"
& $litellm --config $config
