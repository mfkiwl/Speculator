#ifndef FASST_INPUT_PARSER
#define FASST_INPUT_PARSER

#include <string>
#include <vector>
namespace fasst {
	/*!
	This class contains methods to parse input argument list. 
	*/
	class InputParser {
	public:
		/*!
		The main constructor of this class reads input argument and store them into a vector.
		\param argc
		\param argv
		*/
		InputParser(int &argc,char **argv);

		/*!
		This function returns the argument associated to option
		Exemple : 
		if the command is : xxx.exe foo.txt -p -d foo -h
		then, getCmdOption("-d") returns foo
		\param option
		\return argument associated to option, empty string if option does not exist
		*/
		const std::string getCmdOption(const std::string &option) const;

		/*!
		This function returns the argument at index
		Exemple :
		if the command is : xxx.exe foo.txt -p -d foo -h
		then, getArg(1) returns foo.txt
		\param index
		\return argument at index, empty string if index is out of bounds
		*/
		const std::string getArg(const int &index);

		/*!
		This function checks if an option exists
		Exemple :
		if the command is : xxx.exe foo.txt -p -d foo -h
		then, cmdOptionExists("-p") return true
		\param option
		\return true if option appears in argument list, false otherwise
		*/
		bool cmdOptionExists(const std::string &option) const;
	private:
		std::vector <std::string> m_tokens;
	};
}
#endif
