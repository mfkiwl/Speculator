// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#include "../fasst/Source.h"
#include "gtest/gtest.h"

using namespace std;
using namespace fasst;
using namespace tinyxml2;

TEST(Source, Simplest) {
  const char* str = "<source name='first'>"
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
                "<cols>1</cols>"
                "<data>1 </data>"
                "</Hex>"
                "</source>";

  XMLDocument xmlDoc;
  ASSERT_TRUE(xmlDoc.Parse(str) == XML_SUCCESS);

  Source src(xmlDoc.FirstChildElement("source"),0);
  ASSERT_TRUE(src.isInst());
  ASSERT_FALSE(src.isConv());
  ASSERT_EQ(src.rank(), 1);
  ASSERT_EQ(src.wiener_qa(), 1.);
  //ASSERT_EQ(src.wiener_b(), 0.);
  ASSERT_EQ(src.wiener_c1(), 0);
  ASSERT_EQ(src.wiener_c2(), 0);
  ASSERT_EQ(src.wiener_qd(), 0.);
}
