#include "gtest/gtest.h"
#include "../fasst/Audio.h"

#ifdef _WIN32
#include <direct.h>
#define MKDIR(str) _mkdir(str)
#define RMDIR(str) _rmdir(str)
#ifdef _DEBUG
#define CONFIG "Debug"
#elif _RELEASE
#define CONFIG "Release"
#endif

#elif defined(__linux__) || defined(__APPLE__)
#include <cstdlib>
#define MKDIR(str) mkdir(str,S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)
#define RMDIR(str) rmdir(str)
#define CONFIG ""

#endif

// Retrieve global variables declared in gtest main
extern int __argc__;
extern char** __argv__;

using namespace std;
using namespace fasst;
class sourceEstimationTest : public ::testing::Test
{
protected:

    virtual void SetUp()
    {

        ASSERT_GE(__argc__, 3);
        ex = __argv__[1]; // processed example (ex1, ex2 or ex3)
        margin = atof(__argv__[2]);
        cout << "Processing " << ex << " with " << margin << "% of relative error margin" << endl;

        // Create an output directory for current test call
        outDir = "source_estimation_" + ex + "/";
	MKDIR(outDir.c_str());
        string inputAudio = inDir + "/" + ex + "/" + mixName;
        string inputUpSources = inDir + "/" + ex + "/" + updatedSrcName;
        string inputBinDir = inDir + "/" + ex + "/";

        // call exe with input args
        ASSERT_EQ(0, system((exeName + inputAudio + " " + inputUpSources + " " + inputBinDir + " " + outDir).c_str()));
    }
    virtual void TearDown()
    {
        // remove the directory
        EXPECT_EQ(0, RMDIR(outDir.c_str()));
    }

    string ex;
    double margin;
    const string mixName = "mixture.wav"; // mixture file name
    const string RxName = "Rx.bin"; // Rx, computed by comp-rx
    const string RxEnergyName = "Rx_en.bin"; // Rx energy, computed by comp-rx
    const string origSrcName = "sources.xml"; // Original sources
    const string updatedSrcName = "sources.xml.new"; // Updated sources after calling source-estimation
    string binDir = BIN_DIR;
    string mode = CONFIG; // debug or release
    const string inDir = INPUT_DATA_DIR;
    string outDir;
    // Exe 
    const string exeName = binDir + "/" + mode + "/source-estimation ";

};


TEST_F(sourceEstimationTest, objectComparison) {
	// This test compares value by value  :
	// * Generated separated audio sources and groundthruth separated audio source 
    string groundTruthPath = inDir + "/" + ex + "/";
    int J;
    if (ex == "ex1") {
        J = 3; // Number of separated sources
    }
    else {
        J = 2;// Number of separated sources
    }

    for (int j = 1; j <= J; j++) {
        Audio aGround((groundTruthPath + "EstimatedSource_" + to_string(j) + ".wav").c_str());
        Audio aNew((outDir + "EstimatedSource_" + to_string(j) + ".wav").c_str());

        // Compare object equality
        ASSERT_TRUE(aGround.equal(aNew, margin));

        // remove generated EstimatedSource_j.wav
        EXPECT_EQ(0, remove((outDir + "EstimatedSource_" + to_string(j) + ".wav").c_str()));
    }   
    

}
