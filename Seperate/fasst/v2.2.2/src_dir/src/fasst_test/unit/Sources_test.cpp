// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#include "../fasst/Sources.h"
#include "gtest/gtest.h"

#include <stdexcept>

using namespace std;
using namespace fasst;
using namespace tinyxml2;

TEST(Sources, SimpleTest) {
  const char* str = "<sources>"
                "<source>"
                "<A adaptability='free' mixing_type='inst'>"
                "<ndims>2</ndims>"
                "<dim>2</dim>"
                "<dim>1</dim>"
                "<type>real</type>"
                "<data>1 0 </data>"
                "</A>"
                "</source>"
                "<source>"
                "<A adaptability='free' mixing_type='inst'>"
                "<ndims>2</ndims>"
                "<dim>2</dim>"
                "<dim>1</dim>"
                "<type>real</type>"
                "<data>0 1 </data>"
                "</A>"
                "</source>"
                "</sources>";

  XMLDocument xmlDoc;
  ASSERT_TRUE(xmlDoc.Parse(str) == XML_SUCCESS);
  Sources sources(xmlDoc.FirstChildElement("sources"));
  ASSERT_EQ(sources.sources(), 2);
}

TEST(Sources, WrongNumberOfChannels) {
	const char * str = "<sources>"
                "<source>"
                "<A adaptability=\"free\" mixing_type=\"inst\">"
                "<ndims>2</ndims>"
                "<dim>1</dim>"
                "<dim>1</dim>"
                "<type>real</type>"
                "<data>1 </data>"
                "</A>"
                "</source>"
                "<source>"
                "<A adaptability=\"free\" mixing_type=\"inst\">"
                "<ndims>2</ndims>"
                "<dim>2</dim>"
                "<dim>1</dim>"
                "<type>real</type>"
                "<data>1 0 </data>"
                "</A>"
                "</source>"
                "</sources>";

	XMLDocument xmlDoc;
	ASSERT_TRUE(xmlDoc.Parse(str) == XML_SUCCESS);
	ASSERT_THROW(Sources sources(xmlDoc.FirstChildElement("sources")), runtime_error);
}

TEST(Sources, WrongNumberOfBins) {
  const char* str = "<sources>"
                "<source>"
                "<A adaptability=\"free\" mixing_type=\"inst\">"
                "<ndims>2</ndims>"
                "<dim>1</dim>"
                "<dim>1</dim>"
                "<type>real</type>"
                "<data>1 </data>"
                "</A>"
                "<Wex adaptability=\"free\">"
                "<rows>2</rows>"
                "<cols>1</cols>"
                "<data>1 0 </data>"
                "</Wex>"
                "<Uex adaptability=\"fixed\">"
                "<rows>1</rows>"
                "<cols>1</cols>"
                "<data>1 </data>"
                "</Uex>"
                "<Gex adaptability=\"free\">"
                "<rows>1</rows>"
                "<cols>1</cols>"
                "<data>1 </data>"
                "</Gex>"
                "<Hex adaptability=\"fixed\">"
                "<rows>1</rows>"
                "<cols>1</cols>"
                "<data>1 </data>"
                "</Hex>"
                "</source>"
                "<source>"
                "<A adaptability=\"free\" mixing_type=\"inst\">"
                "<ndims>2</ndims>"
                "<dim>1</dim>"
                "<dim>1</dim>"
                "<type>real</type>"
                "<data>0 </data>"
                "</A>"
                "<Wex adaptability=\"free\">"
                "<rows>1</rows>"
                "<cols>1</cols>"
                "<data>1 </data>"
                "</Wex>"
                "<Uex adaptability=\"fixed\">"
                "<rows>1</rows>"
                "<cols>1</cols>"
                "<data>1 </data>"
                "</Uex>"
                "<Gex adaptability=\"free\">"
                "<rows>1</rows>"
                "<cols>1</cols>"
                "<data>1 </data>"
                "</Gex>"
                "<Hex adaptability=\"fixed\">"
                "<rows>1</rows>"
                "<cols>1</cols>"
                "<data>1 </data>"
                "</Hex>"
                "</source>"
                "</sources>";

  XMLDocument xmlDoc;
  ASSERT_TRUE(xmlDoc.Parse(str) == XML_SUCCESS);
  ASSERT_THROW(Sources sources(xmlDoc.FirstChildElement("sources")), runtime_error);
}

TEST(Sources, WrongNumberOfFrames) {
  const char* str = "<sources>"
                "<source>"
                "<A adaptability=\"free\" mixing_type=\"inst\">"
                "<ndims>2</ndims>"
                "<dim>1</dim>"
                "<dim>1</dim>"
                "<type>real</type>"
                "<data>1 </data>"
                "</A>"
                "<Wex adaptability=\"free\">"
                "<rows>1</rows>"
                "<cols>1</cols>"
                "<data>1 </data>"
                "</Wex>"
                "<Uex adaptability=\"fixed\">"
                "<rows>1</rows>"
                "<cols>1</cols>"
                "<data>1 </data>"
                "</Uex>"
                "<Gex adaptability=\"free\">"
                "<rows>1</rows>"
                "<cols>1</cols>"
                "<data>1 </data>"
                "</Gex>"
                "<Hex adaptability=\"fixed\">"
                "<rows>1</rows>"
                "<cols>2</cols>"
                "<data>1\n0 </data>"
                "</Hex>"
                "</source>"
                "<source>"
                "<A adaptability=\"free\" mixing_type=\"inst\">"
                "<ndims>2</ndims>"
                "<dim>1</dim>"
                "<dim>1</dim>"
                "<type>real</type>"
                "<data>0 </data>"
                "</A>"
                "<Wex adaptability=\"free\">"
                "<rows>1</rows>"
                "<cols>1</cols>"
                "<data>1 </data>"
                "</Wex>"
                "<Uex adaptability=\"fixed\">"
                "<rows>1</rows>"
                "<cols>1</cols>"
                "<data>1 </data>"
                "</Uex>"
                "<Gex adaptability=\"free\">"
                "<rows>1</rows>"
                "<cols>1</cols>"
                "<data>1 </data>"
                "</Gex>"
                "<Hex adaptability=\"fixed\">"
                "<rows>1</rows>"
                "<cols>1</cols>"
                "<data>1 </data>"
                "</Hex>"
                "</source>"
                "</sources>";

  XMLDocument xmlDoc;
  ASSERT_TRUE(xmlDoc.Parse(str) == XML_SUCCESS);
  ASSERT_THROW(Sources sources(xmlDoc.FirstChildElement("sources")), runtime_error);
}
