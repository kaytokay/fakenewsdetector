@echo off
setlocal
cd /d "%~dp0"

if /I "%~1"=="start" goto cli_start
if /I "%~1"=="stop" goto cli_stop
if /I "%~1"=="restart" goto cli_restart
if /I "%~1"=="status" goto cli_status
if /I "%~1"=="open" goto cli_open
if /I "%~1"=="help" goto help
if not "%~1"=="" goto help

:menu
cls
echo Guardian-FDT app manager
echo ========================
echo 1. Start and open in browser
echo 2. Stop
echo 3. Restart and open in browser
echo 4. Check status
echo 5. Open in browser
echo 6. Exit
echo.
choice /c 123456 /n /m "Choose an action: "
if errorlevel 6 exit /b 0
if errorlevel 5 goto do_open
if errorlevel 4 goto do_status
if errorlevel 3 goto do_restart
if errorlevel 2 goto do_stop
if errorlevel 1 goto do_start

:do_start
call :StartApp
goto after_action

:do_stop
call :StopApp
goto after_action

:do_restart
call :StopApp
if errorlevel 1 goto after_action
ping 127.0.0.1 -n 3 >nul
call :StartApp
goto after_action

:do_status
call :ShowStatus
goto after_action

:do_open
call :OpenBrowser
goto after_action

:after_action
echo.
pause
goto menu

:cli_start
call :StartApp
exit /b %errorlevel%

:cli_stop
call :StopApp
exit /b %errorlevel%

:cli_restart
call :StopApp
if errorlevel 1 exit /b 1
ping 127.0.0.1 -n 3 >nul
call :StartApp
exit /b %errorlevel%

:cli_status
call :ShowStatus
exit /b %errorlevel%

:cli_open
call :OpenBrowser
exit /b 0

:help
echo Usage: %~nx0 [start^|stop^|restart^|status^|open^|help]
echo Run without arguments to use the interactive menu.
exit /b 0

:StartApp
set "PYTHON_EXE=%~dp0env\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo Creating project virtual environment...
    py -3 -m venv "%~dp0env"
    if errorlevel 1 (
        echo Could not create the virtual environment. Install Python 3.11 or newer and retry.
        exit /b 1
    )
)

call :VerifyLibraries
if errorlevel 1 exit /b 1

call :FindListener
if defined PID (
    echo Guardian-FDT API is already running. Checking model readiness...
    call :WaitForReady
    if errorlevel 1 (
        echo The API did not become ready within 120 seconds. Check its terminal for errors.
        exit /b 1
    )
    echo Guardian-FDT API and models are ready.
    call :OpenBrowser
    exit /b 0
)

start "Guardian-FDT API" cmd /k ""%PYTHON_EXE%" -m uvicorn main:app --host 0.0.0.0 --port 8000"
echo Starting Guardian-FDT API and loading models...
call :WaitForReady
if errorlevel 1 goto startup_timeout
echo Guardian-FDT API and models are ready.
call :OpenBrowser
exit /b 0

:startup_timeout
echo Guardian-FDT API did not become ready within 120 seconds. Check its terminal for errors.
exit /b 1

:StopApp
call :FindListener
if not defined PID (
    echo Guardian-FDT API is not running on port 8000.
    exit /b 0
)
taskkill /PID %PID% /T /F >nul
if errorlevel 1 (
    echo Failed to stop Guardian-FDT API process %PID%.
    exit /b 1
)
echo Stopped Guardian-FDT API.
exit /b 0

:ShowStatus
call :FindListener
if defined PID (
    echo Guardian-FDT API is running on http://localhost:8000 ^(PID %PID%^).
) else (
    echo Guardian-FDT API is not running on port 8000.
)
exit /b 0

:OpenBrowser
start "" http://localhost:8000
exit /b 0

:WaitForReady
set /a WAITED=0
:wait_for_ready
curl.exe --silent --fail --output NUL --max-time 2 http://127.0.0.1:8000/health/ready >nul 2>&1
if not errorlevel 1 exit /b 0
if %WAITED% GEQ 120 exit /b 1
ping 127.0.0.1 -n 2 >nul
set /a WAITED+=1
goto wait_for_ready

:FindListener
set "PID="
for /f "tokens=5" %%A in ('netstat -ano -p tcp ^| findstr /R /C:":8000 .*LISTENING"') do if not defined PID set "PID=%%A"
exit /b 0

:VerifyLibraries
echo Checking required Python libraries and installing missing or incompatible packages...
"%PYTHON_EXE%" -m pip --version >nul 2>&1
if errorlevel 1 (
    echo Python pip is unavailable. Run: "%PYTHON_EXE%" -m ensurepip --upgrade
    exit /b 1
)
"%PYTHON_EXE%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo Library verification or installation failed. Check your internet connection and requirements.txt.
    exit /b 1
)
exit /b 0
