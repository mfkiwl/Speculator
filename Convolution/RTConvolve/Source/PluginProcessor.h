/*
  ==============================================================================

    This file was auto-generated!

    It contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#ifndef PLUGINPROCESSOR_H_INCLUDED
#define PLUGINPROCESSOR_H_INCLUDED

#include "../JuceLibraryCode/JuceHeader.h"
#include "UniformPartitionConvolver.h"
#include "TimeDistributedFFTConvolver.h"
#include "ConvolutionManager.h"

//==============================================================================
/**
*/
class RtconvolveAudioProcessor  : public AudioProcessor
{
public:
    //==============================================================================
    RtconvolveAudioProcessor();
    ~RtconvolveAudioProcessor();

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

   #ifndef JucePlugin_PreferredChannelConfigurations
    bool setPreferredBusArrangement (bool isInput, int bus, const AudioChannelSet& preferredSet) override;
   #endif

    void processBlock (AudioSampleBuffer&, MidiBuffer&) override;

    //==============================================================================
    AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const String getProgramName (int index) override;
    void changeProgramName (int index, const String& newName) override;

    //==============================================================================
    void getStateInformation (MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    //================= CUSTOM =======================
    void setImpulseResponse(const AudioSampleBuffer& impulseResponseBuffer, const juce::String pathToImpulse = "");
private:
//    juce::ScopedPointer<ConvolutionManager<float> > mConvolutionManager[2];
    ConvolutionManager<float> mConvolutionManager[2];
    juce::CriticalSection mLoadingLock;
    float mSampleRate;
    int mBufferSize;
    juce::String mImpulseResponseFilePath;
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (RtconvolveAudioProcessor)
};


#endif  // PLUGINPROCESSOR_H_INCLUDED
