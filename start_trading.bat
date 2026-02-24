@echo off
title Trading Bot Launcher

echo [1/3] Starting Trading Bot...
start cmd /k "python trading_bot.py"

echo [2/3] Starting Backend API Server...
start cmd /k "python server.py"

echo [3/3] Starting React Dashboard...
cd dashboard
start cmd /k "npm start"

echo All systems launching! You can close this window.
pause