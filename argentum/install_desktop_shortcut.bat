@echo off
chcp 65001 >nul 2>&1
title Argentum - Install Desktop Shortcut

echo ============================================================
echo   ARGENTUM - Installing Desktop Shortcut
echo ============================================================
echo.
echo This will create a shortcut "Argentum" on your Desktop.
echo Double-click it any time to launch the trading assistant.
echo.
pause

REM Create temp VBS that builds the shortcut
set "VBS=%TEMP%\argentum_shortcut.vbs"
> "%VBS%" echo Set ws = CreateObject("WScript.Shell")
>> "%VBS%" echo desktop = ws.SpecialFolders("Desktop")
>> "%VBS%" echo Set sc = ws.CreateShortcut(desktop ^& "\Argentum.lnk")
>> "%VBS%" echo sc.TargetPath = "%~dp0start_all.bat"
>> "%VBS%" echo sc.WorkingDirectory = "%~dp0"
>> "%VBS%" echo sc.IconLocation = "imageres.dll,76"
>> "%VBS%" echo sc.Description = "Argentum - AI silver trading assistant"
>> "%VBS%" echo sc.Save
>> "%VBS%" echo WScript.Echo "Created: " ^& desktop ^& "\Argentum.lnk"

cscript //nologo "%VBS%"
del "%VBS%"

echo.
echo Done! Check your Desktop for the "Argentum" shortcut.
echo.
pause
