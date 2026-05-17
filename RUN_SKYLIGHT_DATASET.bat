@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if not exist logs mkdir logs
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set RUN_TS=%%i
set LOG_PATH=logs\skylight_dataset_%RUN_TS%.log

echo Starting skylight dataset runner...
echo Repo: %CD%
echo Log: %LOG_PATH%
echo Queue: data\dataset_queue.csv
echo.

where conda >nul 2>nul
if errorlevel 1 (
  echo ERROR: conda was not found on PATH. >> "%LOG_PATH%"
  echo ERROR: conda was not found on PATH.
  pause
  exit /b 1
)

echo Command: conda run -n lunar python scripts\build_skylight_dataset.py %* >> "%LOG_PATH%"
conda run -n lunar python scripts\build_skylight_dataset.py %* 2>&1 | powershell -NoProfile -Command "$input | Tee-Object -FilePath '%LOG_PATH%' -Append"
set EXIT_CODE=%ERRORLEVEL%

echo.
echo Finished with exit code %EXIT_CODE%.
echo Log: %LOG_PATH%
echo Queue: data\dataset_queue.csv
echo.
pause
exit /b %EXIT_CODE%
