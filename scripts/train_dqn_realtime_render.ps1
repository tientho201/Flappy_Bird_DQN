# Flappy Bird AI Training - Realtime Rendering
Write-Host "Flappy Bird AI Training - Realtime Rendering" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""

# Kiểm tra dependencies
Write-Host "Checking dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host ""
Write-Host "🎮 REALTIME TRAINING MODE" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This mode will show the AI learning in real-time!" -ForegroundColor White
Write-Host "You can watch the AI improve episode by episode." -ForegroundColor White
Write-Host ""
Write-Host "Controls during training:" -ForegroundColor Magenta
Write-Host "- ESC: Quit training" -ForegroundColor White
Write-Host "- R: Reset current episode" -ForegroundColor White
Write-Host "- SPACE: Toggle manual/auto mode" -ForegroundColor White
Write-Host ""
Write-Host "The AI will start randomly and gradually learn to play better!" -ForegroundColor Yellow
Write-Host ""

$fps = Read-Host "Enter FPS for rendering (default 30, press Enter for default)"
if ([string]::IsNullOrEmpty($fps)) { $fps = 30 }

$save_freq = Read-Host "Save model every N episodes (default 50, press Enter for default)"
if ([string]::IsNullOrEmpty($save_freq)) { $save_freq = 50 }

Write-Host ""
Write-Host "Starting realtime training..." -ForegroundColor Green
Write-Host "FPS: $fps" -ForegroundColor Cyan
Write-Host "Save frequency: every $save_freq episodes" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to start training..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

python -m training.train_dqn_realtime_render --config configs/dqn.yaml --fps $fps --save_freq $save_freq

Write-Host ""
Write-Host "Training completed! Check the following:" -ForegroundColor Green
Write-Host "- data/checkpoints/ for saved models" -ForegroundColor White
Write-Host "- logs/ for training metrics" -ForegroundColor White
Write-Host "- Use test.py to test the trained model" -ForegroundColor White
