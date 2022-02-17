@echo off
cmd npm /k "conda config --set ssl_verify False && conda env create -f environment.yaml"