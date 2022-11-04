// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#include "../fasst/XMLDoc.h"
#include "../fasst/Sources.h"
#include "gtest/gtest.h"

#include <stdexcept>

using namespace fasst;
using namespace tinyxml2;
using namespace std;

class XMLDocTest : public ::testing::Test
{
protected:
	
	virtual void SetUp()
	{
		xml1 = 
"<?xml version=""1.0"" encoding=""utf-8""?>"
"<sources>"
"<iterations>200</iterations>"
"<tfr_type>STFT</tfr_type>"
"<wlen>1024</wlen>"
"<source name='first'>"
"<wiener>"
"<a>0</a>"
"<b>0</b>"
"<c1>0</c1>"
"<c2>0</c2>"
"<d>-Inf</d>"
"</wiener>"
"<A adaptability='fixed' mixing_type='inst'>"
"<ndims>2</ndims>"
"<dim>2</dim>"
"<dim>1</dim>"
"<type>real</type>"
"<data>0.382683 0.92388 </data>"
"</A>"
"<Wex adaptability='fixed'>"
"<rows>5</rows>"
"<cols>3</cols>"
"<data>1.1 1.2 1.3 1.4 1.5 \n"
"1.6 1.7 1.8 1.9 1.10 \n"
"1.11 1.12 1.13 1.14 1.15 \n"
"</data>"
"</Wex>"
"<Uex adaptability='fixed'>"
"<rows>4</rows>"
"<cols>4</cols>"
"<data>eye</data>"
"</Uex>"
"<Gex adaptability='fixed'>"
"<rows>4</rows>"
"<cols>4</cols>"
"<data>eye</data>"
"</Gex>"
"<Hex adaptability='fixed'>"
"<rows>3</rows>"
"<cols>4</cols>"
"<data>2.1 2.2 2.3 \n"
"2.4 2.5 2.6 \n"
"2.7 2.8 2.9 \n"
"2.10 2.11 2.12 \n"
"</data>"
"</Hex>"
"</source>"
"<source name='second'>"
"<wiener>"
"<a>0</a>"
"<b>0</b>"
"<c1>0</c1>"
"<c2>0</c2>"
"<d>-Inf</d>"
"</wiener>"
"<A adaptability='free' mixing_type='inst'>"
"<ndims>2</ndims>"
"<dim>2</dim>"
"<dim>1</dim>"
"<type>real</type>"
"<data>0.707107 0.707107 </data>"
"</A>"
"<Wex adaptability='free'>"
"<rows>5</rows>"
"<cols>3</cols>"
"<data>3.1 3.2 3.3 3.4 3.5 \n"
"3.6 3.7 3.8 3.9 3.10 \n"
"3.11 3.12 3.13 3.14 3.15 \n"
"</data>"
"</Wex>"
"<Uex adaptability='fixed'>"
"<rows>4</rows>"
"<cols>4</cols>"
"<data>eye</data>"
"</Uex>"
"<Gex adaptability='fixed'>"
"<rows>4</rows>"
"<cols>4</cols>"
"<data>eye</data>"
"</Gex>"
"<Hex adaptability='free'>"
"<rows>3</rows>"
"<cols>4</cols>"
"<data>4.1 4.2 4.3 \n"
"4.4 4.5 4.6 \n"
"4.7 4.8 4.9 \n"
"4.10 4.11 4.12 \n"
"</data>"
"</Hex>"
"</source>"
"<source name='third'>"
"<wiener>"
"<a>0</a>"
"<b>0</b>"
"<c1>0</c1>"
"<c2>0</c2>"
"<d>-Inf</d>"
"</wiener>"
"<A adaptability='free' mixing_type='inst'>"
"<ndims>2</ndims>"
"<dim>2</dim>"
"<dim>1</dim>"
"<type>real</type>"
"<data>0.92388 0.382683 </data>"
"</A>"
"<Wex adaptability='free'>"
"<rows>5</rows>"
"<cols>3</cols>"
"<data>5.1 5.2 5.3 5.4 5.5 \n"
"5.6 5.7 5.8 5.9 5.10 \n"
"5.11 5.12 5.13 5.14 5.15 \n"
"</data>"
"</Wex>"
"<Uex adaptability='fixed'>"
"<rows>4</rows>"
"<cols>4</cols>"
"<data>eye</data>"
"</Uex>"
"<Gex adaptability='fixed'>"
"<rows>4</rows>"
"<cols>4</cols>"
"<data>eye</data>"
"</Gex>"
"<Hex adaptability='free'>"
"<rows>3</rows>"
"<cols>4</cols>"
"<data>6.1 6.2 6.3 \n"
"6.4 6.5 6.6 \n"
"6.7 6.8 6.8 \n"
"6.9 6.10 6.11 \n"
"</data>"
"</Hex>"
"</source>"
"</sources>";

xml2 =
"<?xml version=""1.0"" encoding=""utf-8""?>"
"<sources>"
"<iterations>200</iterations>"
"<tfr_type>STFT</tfr_type>"
"<wlen>1024</wlen>"
"<source name='first'>"
"<wiener>"
"<a>0</a>"
"<b>0</b>"
"<c1>0</c1>"
"<c2>0</c2>"
"<d>-Inf</d>"
"</wiener>"
"<A adaptability='fixed' mixing_type='inst'>"
"<ndims>2</ndims>"
"<dim>2</dim>"
"<dim>1</dim>"
"<type>real</type>"
"<data>0.5 0.5 </data>"
"</A>"
"<Wex adaptability='fixed'>"
"<rows>5</rows>"
"<cols>3</cols>"
"<data>11.1 11.2 11.3 11.4 11.5 \n"
"11.6 11.7 11.8 11.9 11.10 \n"
"11.11 11.12 11.13 11.14 11.15 \n"
"</data>"
"</Wex>"
"<Uex adaptability='fixed'>"
"<rows>4</rows>"
"<cols>4</cols>"
"<data>eye</data>"
"</Uex>"
"<Gex adaptability='fixed'>"
"<rows>4</rows>"
"<cols>4</cols>"
"<data>eye</data>"
"</Gex>"
"<Hex adaptability='fixed'>"
"<rows>3</rows>"
"<cols>4</cols>"
"<data>22.1 22.2 22.3 \n"
"22.4 22.5 22.6 \n"
"22.7 22.8 22.9 \n"
"22.10 22.11 22.12 \n"
"</data>"
"</Hex>"
"</source>"
"<source name='second'>"
"<wiener>"
"<a>0</a>"
"<b>0</b>"
"<c1>0</c1>"
"<c2>0</c2>"
"<d>-Inf</d>"
"</wiener>"
"<A adaptability='free' mixing_type='inst'>"
"<ndims>2</ndims>"
"<dim>2</dim>"
"<dim>1</dim>"
"<type>real</type>"
"<data>0.6 0.6 </data>"
"</A>"
"<Wex adaptability='free'>"
"<rows>5</rows>"
"<cols>3</cols>"
"<data>33.1 33.2 33.3 33.4 33.5 \n"
"33.6 33.7 33.8 33.9 33.10 \n"
"33.11 33.12 33.13 33.14 33.15 \n"
"</data>"
"</Wex>"
"<Uex adaptability='fixed'>"
"<rows>4</rows>"
"<cols>4</cols>"
"<data>eye</data>"
"</Uex>"
"<Gex adaptability='fixed'>"
"<rows>4</rows>"
"<cols>4</cols>"
"<data>eye</data>"
"</Gex>"
"<Hex adaptability='free'>"
"<rows>3</rows>"
"<cols>4</cols>"
"<data>44.1 44.2 44.3 \n"
"44.4 44.5 44.6 \n"
"44.7 44.8 44.9 \n"
"44.10 44.11 44.12 \n"
"</data>"
"</Hex>"
"</source>"
"<source name='third'>"
"<wiener>"
"<a>0</a>"
"<b>0</b>"
"<c1>0</c1>"
"<c2>0</c2>"
"<d>-Inf</d>"
"</wiener>"
"<A adaptability='free' mixing_type='inst'>"
"<ndims>2</ndims>"
"<dim>2</dim>"
"<dim>1</dim>"
"<type>real</type>"
"<data>0.7 0.7 </data>"
"</A>"
"<Wex adaptability='free'>"
"<rows>5</rows>"
"<cols>3</cols>"
"<data>55.1 55.2 55.3 55.4 55.5 \n"
"55.6 55.7 55.8 55.9 55.10 \n"
"55.11 55.12 55.13 55.14 55.15 \n"
"</data>"
"</Wex>"
"<Uex adaptability='fixed'>"
"<rows>4</rows>"
"<cols>4</cols>"
"<data>eye</data>"
"</Uex>"
"<Gex adaptability='fixed'>"
"<rows>4</rows>"
"<cols>4</cols>"
"<data>eye</data>"
"</Gex>"
"<Hex adaptability='free'>"
"<rows>3</rows>"
"<cols>4</cols>"
"<data>66.1 66.2 66.3 \n"
"66.4 66.5 66.6 \n"
"66.7 66.8 66.8 \n"
"66.9 66.10 66.11 \n"
"</data>"
"</Hex>"
"</source>"
"</sources>";

xml1Name = "test.xml";
xml2Name = "test_updated.xml";
	}

	virtual void TearDown()
	{
		
	}

	const char* xml1;
	const char* xml2;
	const char* xml1Name;
	const char* xml2Name;
};

TEST_F(XMLDocTest, parseXML)
{
	XMLDocument xmlDoc;
	XMLError xmlErr = xmlDoc.Parse(xml1);
	ASSERT_TRUE(xmlErr == XML_SUCCESS);
}

TEST_F(XMLDocTest, writeXML)
{
	XMLDocument xmlDoc;
	xmlDoc.Parse(xml1);
	XMLError xmlErr = xmlDoc.SaveFile(xml1Name);
	ASSERT_TRUE(xmlErr == XML_SUCCESS);
}

TEST_F(XMLDocTest, loadUnexistingXML)
{
	ASSERT_THROW(XMLDoc xmlDoc("this_file_not_exist.xml"), runtime_error);
}

TEST_F(XMLDocTest, loadExistingXML)
{
	ASSERT_NO_THROW(XMLDoc xmlDoc(xml1Name));
}

TEST_F(XMLDocTest, getIters)
{
	XMLDoc xmlDoc(xml1Name);
	int nIters = xmlDoc.getIterations();
	ASSERT_EQ(nIters,200);
}

TEST_F(XMLDocTest, getTFRType)
{
	XMLDoc xmlDoc(xml1Name);
	std::string tfr = xmlDoc.getTFRType();
	ASSERT_EQ(tfr, "STFT");
}

TEST_F(XMLDocTest, getWlen)
{
	XMLDoc xmlDoc(xml1Name);
	int wlen = xmlDoc.getWlen();
	ASSERT_EQ(wlen, 1024);
}

TEST_F(XMLDocTest, getNbin)
{
	XMLDoc xmlDoc(xml1Name);
	int nBins = xmlDoc.getNbin();
	ASSERT_EQ(nBins, 0);
}

TEST_F(XMLDocTest, getSrc)
{
	XMLDoc xmlDoc(xml1Name);
	ASSERT_NO_THROW(Sources s = xmlDoc.getSources());
}

TEST_F(XMLDocTest, replaceSrc)
{
	// This test replace sources in xml1 with updtated ones in xml2
	XMLDoc xmlDoc1(xml1Name); // load from disk

	XMLDocument xmlDocTemp;
	xmlDocTemp.Parse(xml2);
	xmlDocTemp.SaveFile(xml2Name);

	XMLDoc xmlDoc2(xml2Name); // load from disk

	// get two different sources
	fasst::Sources src1init = xmlDoc1.getSources();
	fasst::Sources src2 = xmlDoc2.getSources();

	// Replacement : update xml
	xmlDoc1.updateXML(src2);
    

	xmlDoc1.write("test_after_update.xml");
    fasst::Sources src1updated = xmlDoc1.getSources();
    cout << src1updated[0].A(0) << endl;

	// Check number of sources
	int nSrc1 = src1updated.sources();
	int nSrc2 = src2.sources();
	ASSERT_EQ(nSrc1, nSrc2);

	for (int i = 0; i < nSrc1; i++){
		// Check A matrices for each source (we assume inst mixture (=> A(0)) + real data)	
		ASSERT_EQ(src1updated[i].A(0).rows(), src2[i].A(0).rows());
		ASSERT_EQ(src1updated[i].A(0).cols(), src2[i].A(0).cols());
		for (int j = 0; j < src1updated[i].A(0).rows(); j++){
            for (int k = 0; k < src1updated[i].A(0).cols(); k++){
                if (src1updated[i].A().isFree()) {
                    ASSERT_EQ(src1updated[i].A().getVal(0, j, k).real(), src2[i].A().getVal(0, j, k).real());
                }
                else {
                    ASSERT_EQ(src1updated[i].A().getVal(0, j, k).real(), src1init[i].A().getVal(0, j, k).real());
                }
			}
		}

		// Check Wex matrices (free matrices)
		// Check number of rows
		int src1Wex_nRows = static_cast<int>(src1updated[i].getSPEx().getW().rows());
		int src2Wex_nRows = static_cast<int>(src2[i].getSPEx().getW().rows());
		ASSERT_EQ(src1Wex_nRows, src2Wex_nRows);
		int src1Wex_nCols = static_cast<int>(src1updated[i].getSPEx().getW().cols());
		int src2Wex_nCols = static_cast<int>(src1updated[i].getSPEx().getW().cols());
		ASSERT_EQ(src1Wex_nCols, src2Wex_nCols);
		// Check Wex values
		for (int j = 0; j < src1Wex_nRows; j++){
			for (int k = 0; k < src1Wex_nCols; k++){
                if (src1updated[i].A().isFree()) {
                    ASSERT_EQ(src1updated[i].getSPEx().getW().getVal(j, k), src2[i].getSPEx().getW().getVal(j, k));
                }
                else {
                    ASSERT_EQ(src1updated[i].getSPEx().getW().getVal(j, k), src1init[i].getSPEx().getW().getVal(j, k));
                }
			}
		}

	}
}
