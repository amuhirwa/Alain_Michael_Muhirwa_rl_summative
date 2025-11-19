# Batch Training Script - Train all algorithms with all configurations
# WARNING: This will take several hours to complete!

param(
    [switch]$Quick,
    [switch]$Full
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "CROWD CONTROL RL - BATCH TRAINING SCRIPT" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Determine training mode
if ($Quick) {
    $dqnSteps = 10000
    $ppoSteps = 20000
    $a2cSteps = 15000
    $reinforceEpisodes = 100
    $mode = "QUICK TEST"
    Write-Host "Mode: QUICK TEST (reduced timesteps)" -ForegroundColor Yellow
    Write-Host "Estimated time: 30-60 minutes" -ForegroundColor Gray
} elseif ($Full) {
    $dqnSteps = 100000
    $ppoSteps = 200000
    $a2cSteps = 150000
    $reinforceEpisodes = 1000
    $mode = "FULL TRAINING"
    Write-Host "Mode: FULL TRAINING (assignment-ready)" -ForegroundColor Yellow
    Write-Host "Estimated time: 4-8 hours" -ForegroundColor Gray
} else {
    Write-Host "Please specify training mode:" -ForegroundColor Red
    Write-Host "  Quick test:     .\train_all.ps1 -Quick" -ForegroundColor White
    Write-Host "  Full training:  .\train_all.ps1 -Full" -ForegroundColor White
    Write-Host ""
    exit 1
}

Write-Host ""
$confirmation = Read-Host "Continue with $mode? (yes/no)"
if ($confirmation -ne "yes") {
    Write-Host "Training cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
$startTime = Get-Date

# Train DQN
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "1/4 - TRAINING DQN" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Timesteps: $dqnSteps per configuration (12 configs)" -ForegroundColor Gray
python training/dqn_training.py --timesteps $dqnSteps

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ DQN training complete" -ForegroundColor Green
} else {
    Write-Host "✗ DQN training failed" -ForegroundColor Red
}

# Train PPO
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "2/4 - TRAINING PPO" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Timesteps: $ppoSteps per configuration (12 configs)" -ForegroundColor Gray
python training/ppo_training.py --timesteps $ppoSteps

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ PPO training complete" -ForegroundColor Green
} else {
    Write-Host "✗ PPO training failed" -ForegroundColor Red
}

# Train A2C
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "3/4 - TRAINING A2C" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Timesteps: $a2cSteps per configuration (12 configs)" -ForegroundColor Gray
python training/a2c_training.py --timesteps $a2cSteps

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ A2C training complete" -ForegroundColor Green
} else {
    Write-Host "✗ A2C training failed" -ForegroundColor Red
}

# Train REINFORCE
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "4/4 - TRAINING REINFORCE" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Episodes: $reinforceEpisodes per configuration (12 configs)" -ForegroundColor Gray
python training/reinforce_training.py --episodes $reinforceEpisodes

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ REINFORCE training complete" -ForegroundColor Green
} else {
    Write-Host "✗ REINFORCE training failed" -ForegroundColor Red
}

# Generate comparison
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "GENERATING ALGORITHM COMPARISON" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
python compare_algorithms.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Comparison generated" -ForegroundColor Green
} else {
    Write-Host "✗ Comparison generation failed" -ForegroundColor Red
}

# Calculate total time
$endTime = Get-Date
$duration = $endTime - $startTime
$hours = [math]::Floor($duration.TotalHours)
$minutes = $duration.Minutes

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "✓ ALL TRAINING COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Total time: $hours hours $minutes minutes" -ForegroundColor White
Write-Host ""
Write-Host "Results saved in:" -ForegroundColor Cyan
Write-Host "  - models/dqn/all_results.json" -ForegroundColor White
Write-Host "  - models/ppo/all_results.json" -ForegroundColor White
Write-Host "  - models/a2c/all_results.json" -ForegroundColor White
Write-Host "  - models/reinforce/all_results.json" -ForegroundColor White
Write-Host "  - algorithm_comparison_master.png" -ForegroundColor White
Write-Host "  - algorithm_comparison_summary.csv" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Review results: cat algorithm_comparison_summary.csv" -ForegroundColor White
Write-Host "  2. Run best model: python main.py --algorithm ppo --best --episodes 5" -ForegroundColor White
Write-Host "  3. Compare all: python main.py --compare" -ForegroundColor White
Write-Host "  4. Record video for submission" -ForegroundColor White
Write-Host ""
