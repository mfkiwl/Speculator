ECHO Building PhaseVocoder version number %Version% 
ECHO Build number is %BUILD_NUMBER%

call "%VisualStudioPath%\vcvarsall.bat" amd64

REM Switch to the tag
git checkout tags/v%Version%

REM Run CMake
echo "Creating build dir"
mkdir PhaseVocoderBuilt
echo "Accessing build dir"
cd PhaseVocoderBuilt
echo "Running cmake"
cmake -D VERSION_NUMBER=%Version% -D BUILD_NUMBER=%BUILD_NUMBER% -G "%CMakeGeneratorString%" ..\Source
IF %ERRORLEVEL% NEQ 0 (
    ECHO CMake failed
    EXIT 1
)

REM Do the build
echo "Building release version of PhaseVocoder"
devenv PhaseVocoder.sln /Build Release
IF %ERRORLEVEL% NEQ 0 (
    ECHO The build failed    
    EXIT 1
)

REM Zip the executable
echo "7z" a -tzip PhaseVocoder.zip "%WORKSPACE%\PhaseVocoderBuilt\Application\Release\PhaseVocoder.exe" 
"7z" a -tzip PhaseVocoder.zip "%WORKSPACE%\PhaseVocoderBuilt\Application\Release\PhaseVocoder.exe" 

IF %ERRORLEVEL% NEQ 0 (
    ECHO Zipping PhaseVocoder.exe failed
    EXIT 1
)

REM Copy the resulting installer to the Releases folder
echo copy PhaseVocoder.zip %ReleaseDestination%\PhaseVocoder-Win64-%Version%-%BUILD_NUMBER%.zip
copy PhaseVocoder.zip %ReleaseDestination%\PhaseVocoder-Win64-%Version%-%BUILD_NUMBER%.zip

IF %ERRORLEVEL% NEQ 0 (
    ECHO Copying PhaseVocoder.exe to release destination 
    EXIT 1
)

