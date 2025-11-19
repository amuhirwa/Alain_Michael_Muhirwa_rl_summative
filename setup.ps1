# Automated Setup Script for Crowd Control RL Project
# Run this script to set up everything automatically

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "CROWD CONTROL RL PROJECT - AUTOMATED SETUP" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "  Found: $pythonVersion" -ForegroundColor Green

# Create virtual environment
Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "  Virtual environment already exists, skipping..." -ForegroundColor Gray
} else {
    uv init
    Write-Host "  ✓ Virtual environment created" -ForegroundColor Green
}

# Install dependencies
Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes..." -ForegroundColor Gray
uv add -r requirements.txt --quiet

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ All dependencies installed" -ForegroundColor Green
} else {
    Write-Host "  ✗ Error installing dependencies" -ForegroundColor Red
    Write-Host "  Try running: pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# Run setup test
Write-Host "`nRunning setup tests..." -ForegroundColor Yellow
uv run test_setup.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n============================================================" -ForegroundColor Green
    Write-Host "SETUP COMPLETE - PROJECT READY!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "  1. Test demo: python demo_random_agent.py" -ForegroundColor White
    Write-Host "  2. Quick train: python training/ppo_training.py --config 0 --timesteps 10000" -ForegroundColor White
    Write-Host "  3. Full train: python training/ppo_training.py --timesteps 200000" -ForegroundColor White
    Write-Host "  4. Run model: python main.py --algorithm ppo --best --episodes 5" -ForegroundColor White
    Write-Host "`nFor more info, see QUICKSTART.md" -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host "`n============================================================" -ForegroundColor Red
    Write-Host "SETUP INCOMPLETE - SOME TESTS FAILED" -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host "`nPlease check the errors above and:" -ForegroundColor Yellow
    Write-Host "  1. Ensure Python 3.8+ is installed" -ForegroundColor White
    Write-Host "  2. Try: pip install -r requirements.txt" -ForegroundColor White
    Write-Host "  3. Check QUICKSTART.md for troubleshooting" -ForegroundColor White
    Write-Host ""
}
