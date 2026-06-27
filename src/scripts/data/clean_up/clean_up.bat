@echo off
set ROOT=%~dp0..\..\..\..
del /q "%ROOT%\src\data\*"
for /d %%i in ("%ROOT%\src\data\*") do rmdir /s /q "%%i"
