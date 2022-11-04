#include "InputParser.h"
#include <algorithm>

namespace fasst {
	InputParser::InputParser(int &argc, char **argv) {
		for (int i = 0; i < argc; ++i)
			this->m_tokens.push_back(std::string(argv[i]));
	}

	const std::string InputParser::getCmdOption(const std::string &option) const {
		std::vector<std::string>::const_iterator itr;
		itr = std::find(this->m_tokens.begin(), this->m_tokens.end(), option);
		if (itr != this->m_tokens.end() && ++itr != this->m_tokens.end()) {
			return *itr;
		}
		static const std::string empty_string("");
		return empty_string;
	}

	const std::string InputParser::getArg(const int &index) {
        if (index > 0 && index < static_cast<int>(this->m_tokens.size())) {
			return m_tokens[index];
		}
		else {
			static const std::string empty_string("");
			return empty_string;
		}

	}

	bool InputParser::cmdOptionExists(const std::string &option) const {
		return std::find(this->m_tokens.begin(), this->m_tokens.end(), option)
			!= this->m_tokens.end();
	}
}