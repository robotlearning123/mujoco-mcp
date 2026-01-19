@echo off
:: MuJoCo-MCP Installation Script (Windows)
:: This script helps users install MuJoCo-MCP and its dependencies

echo === MuJoCo-MCP Installation Script (Windows) ===
echo.
echo This script will:
echo 1. Check Python environment
echo 2. Install MuJoCo dependencies
echo 3. Install MCP (Model Context Protocol)
echo 4. Install MuJoCo-MCP in development mode
echo.

:: Confirm to continue
set /p response=Continue with installation? [Y/n]
if /i "%response%"=="n" goto :cancel
if /i "%response%"=="no" goto :cancel

:: Get absolute path of script directory
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."

echo === Checking Python Environment ===
:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python not found
    echo Please install Python 3.8 or higher: https://www.python.org/downloads/
    goto :end
)

:: Check Python version
for /f "tokens=2" %%a in ('python --version 2^>^&1') do set "python_version=%%a"
echo Detected Python version: %python_version%

:: Check if pip is installed
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: pip not found
    echo Please install pip: https://pip.pypa.io/en/stable/installation/
    goto :end
)

echo === Creating Virtual Environment ===
echo Recommended to install in virtual environment

:: Ask if creating virtual environment
set /p create_venv=Create virtual environment? [Y/n]
if /i "%create_venv%"=="n" goto :skip_venv
if /i "%create_venv%"=="no" goto :skip_venv

:: Check venv module
python -c "import venv" >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python venv module not installed
    echo Please install venv module first
    goto :end
)

:: Create virtual environment
echo Creating virtual environment at %REPO_ROOT%\venv...
python -m venv "%REPO_ROOT%\venv"

:: Activate virtual environment
echo Activating virtual environment...
call "%REPO_ROOT%\venv\Scripts\activate.bat"

echo Virtual environment created and activated
goto :venv_done

:skip_venv
echo Skipping virtual environment creation

:venv_done

echo === Upgrading pip and Installing wheel ===
python -m pip install --upgrade pip wheel

echo === Installing MuJoCo Dependencies ===
echo Installing MuJoCo...
python -m pip install mujoco>=2.3.0

:: Check if MuJoCo installed successfully
python -c "import mujoco; print(f'MuJoCo {mujoco.__version__} installed')" >nul 2>&1
if %errorlevel% equ 0 (
    echo MuJoCo installed successfully
) else (
    echo Warning: MuJoCo installation may have issues
    echo Please refer to MuJoCo documentation: https://github.com/deepmind/mujoco
)

echo === Installing Model Context Protocol (MCP) ===
python -m pip install model-context-protocol>=0.1.0

:: Check if MCP installed successfully
python -c "import mcp; print('MCP installed')" >nul 2>&1
if %errorlevel% equ 0 (
    echo MCP installed successfully
) else (
    echo Warning: MCP installation may have issues
)

echo === Installing MuJoCo-MCP ===
cd "%REPO_ROOT%"
python -m pip install -e .

:: Check if MuJoCo-MCP installed successfully
python -c "import mujoco_mcp; print('MuJoCo-MCP installed')" >nul 2>&1
if %errorlevel% equ 0 (
    echo MuJoCo-MCP installed successfully
) else (
    echo Warning: MuJoCo-MCP installation may have issues
)

echo === Installing Optional Dependencies ===
:: Ask if installing Anthropic API
set /p install_anthropic=Install Anthropic API for LLM examples? [Y/n]
if /i "%install_anthropic%"=="n" goto :skip_anthropic
if /i "%install_anthropic%"=="no" goto :skip_anthropic

python -m pip install anthropic
echo Anthropic API installed
goto :anthropic_done

:skip_anthropic
echo Skipping Anthropic API installation

:anthropic_done

echo === Running Verification ===
:: Ask if running verification script
set /p run_verify=Run project verification script? [Y/n]
if /i "%run_verify%"=="n" goto :skip_verify
if /i "%run_verify%"=="no" goto :skip_verify

python "%REPO_ROOT%\tools\project_verify.py"
goto :verify_done

:skip_verify
echo Skipping project verification

:verify_done

echo.
echo === Installation Complete ===
echo.
echo To start using MuJoCo-MCP, try running the demo:
echo python %REPO_ROOT%\examples\demo.py
echo.
echo Or run the LLM integration example:
echo python %REPO_ROOT%\examples\comprehensive_llm_example.py
echo.

if /i not "%create_venv%"=="n" if /i not "%create_venv%"=="no" (
    echo Note: Virtual environment is activated. To use in a new command prompt, run:
    echo call "%REPO_ROOT%\venv\Scripts\activate.bat"
)

goto :end

:cancel
echo Installation cancelled

:end
pause 