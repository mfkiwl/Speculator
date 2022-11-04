// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#include "../fasst/NonNegMatrix.h"
#include "gtest/gtest.h"

#include <stdexcept>

using namespace std;
using namespace fasst;
using namespace tinyxml2;

TEST(NonNegMatrix, ReadValues) {
  const char* str = "<mat adaptability=\"free\">"
                "<rows>2</rows>"
                "<cols>3</cols>"
                "<data>1. 2.\n3. 4.\n5. 6. </data>"
                "</mat>";

  XMLDocument xmlDoc;
  ASSERT_TRUE(xmlDoc.Parse(str) == XML_SUCCESS);
  NonNegMatrix mat(xmlDoc.FirstChildElement("mat"));

  ASSERT_EQ(mat.rows(), 2);
  ASSERT_EQ(mat.cols(), 3);
  ASSERT_TRUE(mat.isFree());
  ASSERT_FALSE(mat.isEye());

  ASSERT_EQ(mat(0, 0), 1.);
  ASSERT_EQ(mat(1, 0), 2.);
  ASSERT_EQ(mat(0, 1), 3.);
  ASSERT_EQ(mat(1, 1), 4.);
  ASSERT_EQ(mat(0, 2), 5.);
  ASSERT_EQ(mat(1, 2), 6.);


}
