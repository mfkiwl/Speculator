#!/bin/bash
#This script download and install PEASS backend.
PEASSURL="http://bass-db.gforge.inria.fr/peass/PEASS-Software-v2.0.1.zip"
THIRDPARTYURL="http://medi.uni-oldenburg.de/download/demo/adaption-loops/adapt_loop.zip"
echo "PEASS installation process"
echo ""

#LOCALPATH=$(dirname $0)
if [ "$#" -eq 0 ]
then
LOCALPATH=$PWD
else
if [ "$#" -eq 1 ]
then
LOCALPATH=$1
else
echo "Wrong number of arguments"
return 0
fi
fi

#echo "Local path : $LOCALPATH"
PEASSPATH=$LOCALPATH/PEASS-Software-v2.0.1
#echo "PEASS install path : $PEASSPATH"

download_and_unzip ()
{
  echo "Download and unzip PEASS:"
  # check if directory exist and remove it
  if [ -d "$PEASSPATH" ];
  then
    rm -R $PEASSPATH
  fi

  curl -O $PEASSURL > /dev/null 2>&1
  local stat1=$?
  if [ $stat1 -eq 0 ]
  then
  unzip PEASS-Software-v2.0.1.zip -d $LOCALPATH > /dev/null 2>&1
  rm PEASS-Software-v2.0.1.zip > /dev/null 2>&1
  mv $PEASSPATH/readme.txt $PEASSPATH/readme_peass.txt
  curl -O $THIRDPARTYURL > /dev/null 2>&1
  local stat2=$?
  if [ $stat2 -eq 0 ]
  then
  unzip adapt_loop.zip -d $PEASSPATH > /dev/null 2>&1
  rm adapt_loop.zip
  return 0
else
  return 1
  fi

else
  return 1
  fi
}

check_files () {

  local errorCpt=0
  declare -a list_peass=(
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
"adapt_m.c"
"adapt.h"
"adapt_m.dll"
"adapt.dll"
"adapt.c"
"haircell.c"
"haircell.dll"
"gammatone/Example_Filter.m"
"gammatone/Example_Filterbank.m"
"gammatone/Example_Synthesis.m"
"gammatone/Gfb_Analyzer_clear_state.m"
"gammatone/Gfb_Analyzer_fprocess.c"
"gammatone/Gfb_Analyzer_new.m"
"gammatone/Gfb_Analyzer_process.m"
"gammatone/Gfb_Analyzer_zresponse.m"
"gammatone/Gfb_Delay_clear_state.m"
"gammatone/Gfb_Delay_new.m"
"gammatone/Gfb_Delay_process.m"
"gammatone/Gfb_Filter_clear_state.m"
"gammatone/Gfb_Filter_new.m"
"gammatone/Gfb_Filter_process.m"
"gammatone/Gfb_Filter_zresponse.m"
"gammatone/Gfb_Mixer_new.m"
"gammatone/Gfb_Mixer_process.m"
"gammatone/Gfb_Synthesizer_clear_state.m"
"gammatone/Gfb_Synthesizer_new.m"
"gammatone/Gfb_Synthesizer_process.m"
"gammatone/Gfb_analyze.c"
"gammatone/Gfb_analyze.h"
"gammatone/Gfb_center_frequencies.m"
"gammatone/Gfb_erbscale2hz.m"
"gammatone/Gfb_hz2erbscale.m"
"gammatone/Gfb_plot.m"
"gammatone/Gfb_set_constants.m"
  )
  for i in "${list_peass[@]}"
  do
    if [ ! -f "$PEASSPATH/$i" ]
    then
      ((errorCpt++))
    fi
  done

echo "    - Number of missing file(s): $errorCpt."
return $errorCpt
}

check_mex () {
  #echo "Looking for mex files"
  local errorCpt=0
  if ls $PEASSPATH/adapt.mex* &> /dev/null; then
    :
  else
    #echo "- adapt mex DOES NOT EXIST"
    ((errorCpt++))
  fi

  if ls $PEASSPATH/haircell.mex* &> /dev/null; then
    :
  else
    #echo "- haircell mex DOES NOT EXIST"
    ((errorCpt++))
  fi

  if ls $PEASSPATH/toeplitzC.mex* &> /dev/null; then
    :
  else
    #echo "- toeplitzC mex DOES NOT EXIST"
    ((errorCpt++))
  fi

  if ls $PEASSPATH/gammatone/Gfb_Analyzer_fprocess.mex* &> /dev/null; then
    :
  else
    #echo "- Gfb_Analyzer_fprocess mex DOES NOT EXIST"
    ((errorCpt++))
  fi

  echo "    - Number of missing mex files: $errorCpt."
  return $errorCpt
}

compile () {
  echo "Compile"
  matlab -nodesktop -nojvm -r "run('$PEASSPATH/compile.m');exit;" > /dev/null 2>&1
  local retCMD=$?
  if [ $retCMD -ne 0 ]
  then
  echo ""
  echo "-------------------------------------------------------------------------------"
  echo "ERROR[2] Unable to find Matlab in your PATH."
  echo "In order to solve this problem, you can find some advices in the QuickStarter file."
  echo "However, PEASS is not a mandatory software to run provided FASST examples."
  echo "-------------------------------------------------------------------------------"
  return 1
  fi
  #echo "End of PEASS compilation"
  check_mex
  local retMex=$?
  if [ $retMex -ne 0 ]
  then
    echo ""
    echo "-------------------------------------------------------------------------------"
    echo "ERROR [3] Unable to compile mex function(s)."
    echo "In order to solve this problem, you can find some advices in the QuickStarter file."
    echo "However, PEASS is not a mandatory software to run provided FASST examples."
    echo "-------------------------------------------------------------------------------"
    return 1
  else
    return 0
  fi

}

update_examples () {
  #echo "Update example files"
  # Update PEASS Status and PEASS flag into example2 and example3
  local OLD_PEASSBACKEND="PEASS_BACK_END = 0"
  local NEW_PEASSBACKEND="PEASS_BACK_END = 1"
  local OLD_PEASSPATH="PEASS_PATH = ''"
  local NEW_PEASSPATH="PEASS_PATH = '$PEASSPATH'"
  for ex in example2
  do
  	sed -i '' "s/$OLD_PEASSBACKEND/$NEW_PEASSBACKEND/" $LOCALPATH/$ex/$ex.m
  	sed -i '' "s|$OLD_PEASSPATH|$NEW_PEASSPATH|g" $LOCALPATH/$ex/$ex.m
  done
  echo ""
  echo "-------------------------------------------------------------------------------"
  echo "PEASS compilation and installation sucessful!"
  echo "-------------------------------------------------------------------------------"
}

### Main script starts here
continueProcessing=1
echo "Check installation status:"
check_files
ret=$?
if [ $ret -ne 0 ]
then
# Download peass and adapt loop
download_and_unzip $LOCALPATH $PEASSPATH
check_files
ret=$?
if [ $ret -ne 0 ]
then
echo ""
echo "-------------------------------------------------------------------------------"
echo "ERROR[1] Unable to download PEASS."
echo "In order to solve this problem, you can find some advices in the QuickStarter file."
echo "However, PEASS is not a mandatory software to run provided FASST examples."
echo "-------------------------------------------------------------------------------"
continueProcessing=0
fi
fi

# check if mex files exist
if [ $continueProcessing -eq 1 ]
then
echo "Check installation status:"
check_mex
ret=$?
if [ $ret -ne 0 ]
then
# Compile mex files
compile
ret=$?
if [ $ret -eq 0 ]
then
# Download peass and adapt loop
update_examples
fi
fi
fi
echo ""
echo "Press a key to continue ..."
read
exit
