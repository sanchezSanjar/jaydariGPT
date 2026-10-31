@echo off
rem Builds JaydariGPT.exe in the project folder using the C# compiler built into Windows.
cd /d "%~dp0"
set CSC=%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe
if not exist "%CSC%" set CSC=%WINDIR%\Microsoft.NET\Framework\v4.0.30319\csc.exe
if not exist "%CSC%" (
  echo .NET Framework 4 C# compiler not found.
  pause
  exit /b 1
)
if not exist "..\assets\jaydarigpt.ico" powershell -NoProfile -ExecutionPolicy Bypass -File make_icon.ps1
"%CSC%" /nologo /target:winexe /optimize+ /win32icon:..\assets\jaydarigpt.ico ^
  /reference:System.Windows.Forms.dll /out:..\JaydariGPT.exe JaydariGPT.cs
if errorlevel 1 (
  echo Build failed.
  pause
  exit /b 1
)
echo Built JaydariGPT.exe
