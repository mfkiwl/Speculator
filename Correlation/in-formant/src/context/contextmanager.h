#ifndef MAIN_CONTEXT_MANAGER_H
#define MAIN_CONTEXT_MANAGER_H

#include "../analysis/analysis.h"
#include "../modules/app/app.h"
#include "audiocontext.h"
#include "rendercontext.h"
#include "guicontext.h"
#include "datastore.h"
#include "views/views.h"
#include "config.h"
#include <atomic>
#include <thread>

#ifdef ENABLE_TORCH
#include "../analysis/formant/deepformants/df.h"
#endif

#include "synthwrapper.h"
#include "dataviswrapper.h"

#define STR_(arg) #arg
#define STR(arg) STR_(arg)
#define INFORMANT_VERSION_STR STR(INFORMANT_VERSION)

#ifdef _WIN32
#include <windows.h>
LONG WINAPI TopLevelExceptionHandler(PEXCEPTION_POINTERS pExceptionInfo);
void StdExceptionHandler(const std::exception& e);
#endif

namespace Main {

    using namespace Module;

    using dur_ms = std::chrono::milliseconds;

    extern int argc;
    extern char **argv;

    class ContextManager;
    extern std::unique_ptr<ContextManager> contextManager;

    class ContextManager {
    public:
        ContextManager(
                int captureSampleRate,
                const dur_ms &playbackBlockDuration,
                int playbackSampleRate);

        int exec();

    private:
        void createViews();
        void loadConfig();
        
        void openAndStartAudioStreams();

        void startAnalysisThread();
        void analysisThreadLoop();
        void stopAnalysisThread();

#ifndef WITHOUT_SYNTH
        void startSynthesisThread();
        void synthesisThreadLoop();
        void stopSynthesisThread();
#endif
        
        void datavisThreadLoop();

        void setView(const std::string &name);

        std::unique_ptr<Config> mConfig;

        std::shared_ptr<Analysis::PitchSolver> mPitchSolver;
        std::shared_ptr<Analysis::LinpredSolver> mLinpredSolver;
        std::shared_ptr<Analysis::FormantSolver> mFormantSolver;
        std::shared_ptr<Analysis::InvglotSolver> mInvglotSolver;

#ifdef ENABLE_TORCH
        DFModelHolder *mDfModelHolder;
#endif
        
        std::unique_ptr<Audio::Buffer> mCaptureBuffer;
        std::unique_ptr<Audio::Queue> mPlaybackQueue;

        std::unique_ptr<DataStore> mDataStore;
        
        std::unique_ptr<App::Pipeline> mPipeline;
       
#ifndef WITHOUT_SYNTH
        std::unique_ptr<App::Synthesizer> mSynthesizer;
        SynthWrapper mSynthWrapper;
#endif
        
        DataVisWrapper mDataVisWrapper;

        std::unique_ptr<AudioContext> mAudioContext;
        std::unique_ptr<RenderContext> mRenderContext;
        std::unique_ptr<GuiContext> mGuiContext;

        rpm::map<std::string, std::unique_ptr<AbstractView>> mViews;
        
        std::thread mAnalysisThread;
        std::atomic_bool mAnalysisRunning;

#ifndef WITHOUT_SYNTH
        std::thread mSynthesisThread;
        std::atomic_bool mSynthesisRunning;
#endif

        std::thread mDatavisThread;

        int mViewMinFrequency;
        int mViewMaxFrequency;
        int mViewFftSize;

        int mAnalysisMaxFrequency;
        int mAnalysisLpOffset;
        int mAnalysisLpOrder;
        int mAnalysisPitchSampleRate;
    };

}

#endif // MAIN_CONTEXT_MANAGER_H
