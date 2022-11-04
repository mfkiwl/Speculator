@echo off
SETLOCAL EnableDelayedExpansion
echo PEASS installation process
echo.
REM option : download / compile
set PEASSPATH=%~dp0PEASS-Software-v2.0.1
set LOCALPATH=%~dp0
set checkErrors=0
set continueProcessing=1
REM Check that all files exist
echo Check installation status
call:CHECK_FILES checkErrors %PEASSPATH%
IF !checkErrors! NEQ 0 (
REM Download PEASS & adapt_loop
call:DOWNLOAD_AND_UNZIP "" %LOCALPATH% %PEASSPATH%
call:CHECK_FILES checkErrors %PEASSPATH%
IF !checkErrors! NEQ 0 (
echo.
echo -------------------------------------------------------------------------------
echo ERROR[1] Unable to download PEASS
echo In order to solve this problem, you can find some advices in the QuickStarter file
echo However, PEASS is not a mandatory software to run provided FASST examples
echo -------------------------------------------------------------------------------
set continueProcessing=0
)
)
IF !continueProcessing! EQU 1 (
REM Check if MEX files exist
echo Check installation status
call:CHECK_MEX checkErrors %PEASSPATH%
IF !checkErrors! NEQ 0 (
REM Compile Mex-files
call:COMPILE checkErrors %PEASSPATH%
IF !checkErrors! EQU 0 (
REM Update example files
call:UPDATE_EXAMPLES "" %LOCALPATH%
)
)
)
echo.
PAUSE
ENDLOCAL
goto:EOF

:DOWNLOAD_AND_UNZIP
REM check if 
SETLOCAL
echo Download and unzip PEASS
IF EXIST %~3 (
REM echo removing dir
RMDIR /S /Q "%~3"
)
REM Download PEASS
powershell -Command "(New-Object Net.WebClient).DownloadFile('http://bass-db.gforge.inria.fr/peass/PEASS-Software-v2.0.1.zip','%TEMP%\peass.zip')" > nul
REM unzip PEASS
powershell -Command "& { $shell = New-Object -COM Shell.Application; $target = $shell.NameSpace('%~2'); $zip = $shell.NameSpace('%TEMP%\peass.zip'); $target.CopyHere($zip.Items(), 16); }" > nul
REM Download Adapt_loop
powershell -Command "(New-Object Net.WebClient).DownloadFile('http://medi.uni-oldenburg.de/download/demo/adaption-loops/adapt_loop.zip','%TEMP%\adapt_loop.zip')" > nul
REM unzip Adapt loops/adapt_loop
powershell -Command "& { $shell = New-Object -COM Shell.Application; $target = $shell.NameSpace('%~3'); $zip = $shell.NameSpace('%TEMP%\adapt_loop.zip'); $target.CopyHere($zip.Items(), 16); }" > nul
DEL %TEMP%\adapt_loop.zip >nul 2>&1
DEL %TEMP%\peass.zip >nul 2>&1
ENDLOCAL
goto:EOF

:COMPILE 
SETLOCAL EnableDelayedExpansion
echo Compile
set "localError_compile=0"
REM WHERE matlab
WHERE matlab >nul 2>&1
IF !ERRORLEVEL! NEQ 0 (
echo.
echo -------------------------------------------------------------------------------
echo ERROR[2] Unable to find Matlab in your PATH
echo In order to solve this problem, you can find some advices in the QuickStarter file
echo However, PEASS is not a mandatory software to run provided FASST examples
echo -------------------------------------------------------------------------------
set "localError_compile=1"
goto:ending
)
IF !ERRORLEVEL! EQU 0 (
start /WAIT matlab -wait -nodesktop -minimize -r "run('%~2\compile.m');exit;"

REM set toto=0
call:CHECK_MEX toto "%~2"

 IF !toto! EQU 0 (
 
 goto:ending
 )
 IF !toto! NEQ 0 (
 echo.
 echo -------------------------------------------------------------------------------
 echo ERROR[3] Unable to compile mex function(s)
 echo In order to solve this problem, you can find some advices in the QuickStarter file
 echo However, PEASS is not a mandatory software to run provided FASST examples
 echo -------------------------------------------------------------------------------
 set "localError_compile=1"
 goto:ending
 ) 
)

:ending
ENDLOCAL&Set %~1=%localError_compile%

goto:EOF

:CHECK_MEX 
SETLOCAL EnableDelayedExpansion
set "localError_mex=0"
IF EXIST "%~2\adapt.mexw*" (
REM echo - Adapt mex DOES EXIST
) ELSE (
REM echo - Adapt mex DOES NOT EXIST
Set /A localError_mex+=1
)

IF EXIST "%~2\haircell.mexw*" (
REM echo - Haircell mex DOES EXIST
) ELSE (
REM echo - Haircell mex DOES NOT EXIST
Set /A localError_mex+=1
)

IF EXIST "%~2\toeplitzC.mexw*" (
REM echo - ToeplitzC mex DOES EXIST
) ELSE (
REM echo - ToeplitzC mex DOES NOT EXIST
Set /A localError_mex+=1
)

IF EXIST "%~2\gammatone\Gfb_Analyzer_fprocess.mexw*" (
REM echo - Gfb_Analyzer_fprocess mex DOES EXIST
) ELSE (
REM echo - Gfb_Analyzer_fprocess mex DOES NOT EXIST
Set /A localError_mex+=1
)
echo     - Number of missing mex files: !localError_mex!
ENDLOCAL&Set %~1=%localError_mex%

goto:EOF

:CHECK_FILES 
SETLOCAL
set "localError_files=0"

for %%a in (
"ISR_SIR_SAR_fromNewDecomposition.m"                   
"LSDecompose.m"                        
"LSDecompose_tv.m"
"map2SubjScale.m"                     
"PEASS_ObjectiveMeasure.m"
"myMapping.m"
"myPemoAnalysisFilterBank.m"
"myPemoSynthesisFilterBank.m"
"paramTask1.mat"        
"paramTask2.mat"
"paramTask3.mat"
"paramTask4.mat"    
"audioQualityFeatures.m"
"pemo_internal.m"             
"compile.m"
"pemo_metric.m"                   
"erbBW.m"
"toeplitzC.c"                     
"example.m"              
"extractDistortionComponents.m"
"extractTSIA.m"
) do (
IF NOT EXIST "%~2\%%a" (
Set /A localError_files+=1
)
)

for %%a in (
"adapt_m.c"
"adapt.h"
"adapt_m.dll"
"adapt.dll"
"adapt.c"
"haircell.c"
"haircell.dll"
"gammatone\Example_Filter.m"
"gammatone\Example_Filterbank.m"
"gammatone\Example_Synthesis.m"
"gammatone\Gfb_Analyzer_clear_state.m"
"gammatone\Gfb_Analyzer_fprocess.c"
"gammatone\Gfb_Analyzer_new.m"
"gammatone\Gfb_Analyzer_process.m"
"gammatone\Gfb_Analyzer_zresponse.m"
"gammatone\Gfb_Delay_clear_state.m"
"gammatone\Gfb_Delay_new.m"
"gammatone\Gfb_Delay_process.m"
"gammatone\Gfb_Filter_clear_state.m"
"gammatone\Gfb_Filter_new.m"
"gammatone\Gfb_Filter_process.m"
"gammatone\Gfb_Filter_zresponse.m"
"gammatone\Gfb_Mixer_new.m"
"gammatone\Gfb_Mixer_process.m"
"gammatone\Gfb_Synthesizer_clear_state.m"
"gammatone\Gfb_Synthesizer_new.m"
"gammatone\Gfb_Synthesizer_process.m"
"gammatone\Gfb_analyze.c"
"gammatone\Gfb_analyze.h"
"gammatone\Gfb_center_frequencies.m"
"gammatone\Gfb_erbscale2hz.m"
"gammatone\Gfb_hz2erbscale.m"
"gammatone\Gfb_plot.m"
"gammatone\Gfb_set_constants.m"
) do (
IF NOT EXIST "%~2\%%a" (
Set /A localError_files+=1
)
)
echo     - Number of missing file(s): %localError_files%
ENDLOCAL&Set %~1=%localError_files%
goto:EOF

:UPDATE_EXAMPLES
SETLOCAL
REM echo Update example files
for %%a in (example2) do (
powershell -Command "&{Get-Content %~2\%%a\%%a.m | %%{$_ -replace 'PEASS_BACK_END = 0', 'PEASS_BACK_END = 1'} | Set-Content %~2\%%a\%%a_new.m}"
COPY /Y %~2\%%a\%%a_new.m %~2\%%a\%%a.m >nul 2>&1
DEL %~2\%%a\%%a_new.m >nul 2>&1
powershell -Command "Get-Content '%~2\%%a\%%a.m' | %%{$_ -replace 'PEASS_PATH = ''''', 'PEASS_PATH = ''.\..\PEASS-Software-v2.0.1\'''} | Set-Content '%~2\%%a\%%a_new.m'"
COPY /Y %~2\%%a\%%a_new.m %~2\%%a\%%a.m >nul 2>&1
DEL %~2\%%a\%%a_new.m >nul 2>&1
)
echo.
echo -------------------------------------------------------------------------------
echo PEASS compilation and installation sucessful!
echo -------------------------------------------------------------------------------
ENDLOCAL
goto:EOF

:EOF