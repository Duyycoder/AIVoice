@echo off
setlocal
rem ============================================================
rem  Train LoRA nhan vat cho AI Storytelling
rem  Cach dung:
rem    train_lora.bat <slug_nhan_vat> <thu_muc_anh> "<instance_prompt>" [so_buoc]
rem  Vi du:
rem    train_lora.bat d_ch_phong "D:\anh\dichphong" "1boy, silver hair with black and yellow streaks, red eyes" 800
rem ============================================================

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if "%~3"=="" (
    echo [LOI] Thieu tham so.
    echo Cach dung: train_lora.bat ^<slug^> ^<thu_muc_anh^> "<instance_prompt>" [so_buoc]
    echo Vi du:     train_lora.bat d_ch_phong "D:\anh\dichphong" "1boy, silver hair, red eyes" 800
    pause
    exit /b 1
)

set "STEPS=%~4"
if "%STEPS%"=="" set "STEPS=800"

set "PYTHONPATH=%SCRIPT_DIR%"
"..\..\.venv\Scripts\python.exe" scripts\train_character_lora.py --character "%~1" --images_dir "%~2" --instance_prompt "%~3" --steps %STEPS%
pause
