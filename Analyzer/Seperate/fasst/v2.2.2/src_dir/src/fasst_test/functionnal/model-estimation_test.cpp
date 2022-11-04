#include "gtest/gtest.h"
#include "../fasst/Sources.h"
#include "../fasst/XMLDoc.h"

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

class modelEstimationTest : public ::testing::Test
{
protected:

    virtual void SetUp()
    {
	ASSERT_GE(__argc__, 3);
        ex = __argv__[1]; // processed example (ex1, ex2 or ex3)
        margin = atof(__argv__[2]);
        cout << "Processing " << ex << " with " << margin << "% of relative error margin" << endl;
        // Create an output directory for current test call
        outDir = "./model_estimation_" + ex + "/";
        MKDIR(outDir.c_str());

        // call exe with input args
        string inputSources = inDir + "/" + ex + "/" + origSrcName;
        string inputBinDir = inDir + "/" + ex + "/";
        ASSERT_EQ(0, system((exeName + inputSources + " " + inputBinDir + " " + outDir + updatedSrcName).c_str()));
    }
    virtual void TearDown()
    {
        EXPECT_EQ(0, remove((outDir + updatedSrcName).c_str()));
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
    const string exeName = binDir + "/" + mode + "/model-estimation ";

};

TEST_F(modelEstimationTest, objectComparison) {
	// This test compares value by value  :
	// * Generated sources model by model-estimation and groundtruth sources model
    string groundTruthPath = inDir + "/" + ex + "/";
    Sources sources_ground(XMLDoc((groundTruthPath + updatedSrcName).c_str()).getSources());
    Sources sources_new(XMLDoc((outDir + updatedSrcName).c_str()).getSources());
    
    ASSERT_TRUE(sources_ground.equal(sources_new, margin));
    
}
