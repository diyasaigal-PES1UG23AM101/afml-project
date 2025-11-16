# Set OpenAI API Key from .env file
# Run this before running tests or the application

$envFile = Join-Path $PSScriptRoot ".env"
$envExample = Join-Path $PSScriptRoot ".env.example"

if (Test-Path $envFile) {
    Write-Host "Loading environment variables from .env file..." -ForegroundColor Green
    
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]*)\s*=\s*(.*)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            
            # Remove quotes if present
            $value = $value -replace '^["'']|["'']$', ''
            
            # Set environment variable
            [Environment]::SetEnvironmentVariable($name, $value, 'Process')
            Write-Host "  Set $name" -ForegroundColor Cyan
        }
    }
    
    Write-Host "`nEnvironment variables loaded successfully!" -ForegroundColor Green
    Write-Host "You can now run: python test_pipeline.py" -ForegroundColor Yellow
    
} elseif (Test-Path $envExample) {
    Write-Host ".env file not found!" -ForegroundColor Red
    Write-Host "`nCreating .env from .env.example..." -ForegroundColor Yellow
    Copy-Item $envExample $envFile
    Write-Host ".env file created!" -ForegroundColor Green
    Write-Host "`nIMPORTANT: Edit .env file and add your actual OpenAI API key" -ForegroundColor Yellow
    Write-Host "Then run this script again: .\setup_env.ps1" -ForegroundColor Yellow
    
} else {
    Write-Host "Neither .env nor .env.example found!" -ForegroundColor Red
    Write-Host "Please create .env file with:" -ForegroundColor Yellow
    Write-Host "OPENAI_API_KEY=your-api-key-here" -ForegroundColor Cyan
}
