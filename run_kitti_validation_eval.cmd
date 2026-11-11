@echo off
call "%~dp0run.cmd" --dataset kitti --split validation --evaluate %*
