/*
 * PhaseVocoder
 *
 * Copyright (c) 2017 - Terence M. Darwen - tmdarwen.com
 *
 * The MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <string>
#include <map>

class CommandLineArguments
{
	public:
		CommandLineArguments(int argc, char** argv);

		bool IsValid() const;

		const std::string& GetErrorMessage() const;

		bool InputFilenameGiven() const;
		const std::string GetInputFilename() const;

		bool OutputFilenameGiven() const;
		const std::string GetOutputFilename() const;

		bool StretchFactorGiven() const;
		double GetStretchFactor() const;

		bool PitchSettingGiven() const;
		double GetPitchSetting() const;

		bool ResampleSettingGiven() const;
		std::size_t GetResampleSetting() const;

		bool ValleyPeakRatioGiven() const;
		double GetValleyPeakRatio() const;

		bool ShowTransients() const;
		bool TransientConfigFileGiven() const;
		const std::string GetTransientConfigFilename() const;
		bool Help() const;
		bool LongHelp() const;
		bool Version() const;

	private:
		bool ParseArguments(int argc, char** argv);
		void ValidateArguments();
		bool ValidateStretchSetting();
		bool ValidatePitchSetting();
		bool ValidateResampleSetting();

		bool valid_{true};
		std::string errorMessage_;

		std::map<std::string, std::string> argumentsGiven_;

		// The stretch factor must be between 0.01 and 10.0
		const double minimumStretchFactor_{0.01};
		const double maximumStretchFactor_{10.0};

		// The pitch shift must be between -24.0 and +24.0 semitones
		const double minimumPitchShift_{-24.0};
		const double maximumPitchShift_{24.0};

		// The stretch factor must be between 0.01 and 10.0
		const std::size_t minimumResampleFrequency_{1000};
		const std::size_t maximumResampleFrequency_{192000};

		struct ArgumentTraits
		{
			ArgumentTraits() : acceptsValue_{false}, requiresValue_{false} { }
			ArgumentTraits(const std::string& shortArgument) : shortArgument_{shortArgument}, acceptsValue_{false}, requiresValue_{false} { }
			ArgumentTraits(const std::string& shortArgument, bool acceptsValue, bool requiresValue) : 
				shortArgument_{shortArgument}, acceptsValue_{acceptsValue}, requiresValue_{requiresValue} { }
			std::string shortArgument_;
			bool acceptsValue_;
			bool requiresValue_;
		};

		std::map<std::string, ArgumentTraits> possibleArguments_;
};