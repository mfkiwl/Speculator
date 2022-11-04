// Copyright 2018 INRIA
// This software is released under the Q Public License Version 1.0.
// https://opensource.org/licenses/QPL-1.0

#include "../fasst/MixingParameter.h"
#include "../fasst/XMLDoc.h"
#include "gtest/gtest.h"

#include <stdexcept>

using namespace std;
using namespace fasst;
using namespace tinyxml2;

TEST(MixingParameter, InstReal) {
	const char* str = "<A adaptability='free' mixing_type='inst'>"
		"<ndims>2</ndims>"
		"<dim>2</dim>"
		"<dim>3</dim>"
		"<type>real</type>"
		"<data>1. 2. 3. 4. 5. 6. </data>"
		"</A>";
	XMLDocument xmlDoc;
	ASSERT_TRUE(xmlDoc.Parse(str) == XML_SUCCESS);
	
	MixingParameter A(xmlDoc.FirstChildElement("A"));

	ASSERT_TRUE(A.isInst());
	ASSERT_FALSE(A.isConv());
	ASSERT_EQ(A.rank(), 3);

  ASSERT_EQ(A.getVal(0, 0, 0).real(), 1.);
  ASSERT_EQ(A.getVal(0, 1, 0).real(), 2.);
  ASSERT_EQ(A.getVal(0, 0, 1).real(), 3.);
  ASSERT_EQ(A.getVal(0, 1, 1).real(), 4.);
  ASSERT_EQ(A.getVal(0, 0, 2).real(), 5.);
  ASSERT_EQ(A.getVal(0, 1, 2).real(), 6.);

  ASSERT_EQ(A(0)(0, 0), 1.);
  ASSERT_EQ(A(0)(1, 0), 2.);
  ASSERT_EQ(A(0)(0, 1), 3.);
  ASSERT_EQ(A(0)(1, 1), 4.);
  ASSERT_EQ(A(0)(0, 2), 5.);
  ASSERT_EQ(A(0)(1, 2), 6.);

}

TEST(MixingParameter, ConvReal) {
	const char* str = "<A adaptability='free' mixing_type='conv'>"
		"<ndims>3</ndims>"
		"<dim>2</dim>"
		"<dim>3</dim>"
		"<dim>4</dim>"
		"<type>real</type>"
		"<data>1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11. 12. 13. 14. 15. 16. 17. 18. "
		"19. 20. 21. 22. 23. 24. </data>"
		"</A>";
	XMLDocument xmlDoc;
	ASSERT_TRUE(xmlDoc.Parse(str) == XML_SUCCESS);
	MixingParameter A(xmlDoc.FirstChildElement("A"));

	ASSERT_FALSE(A.isInst());
	ASSERT_TRUE(A.isConv());
	ASSERT_EQ(A.rank(), 3);
	ASSERT_EQ(A.size(), 4); // nb of freq bins

  ASSERT_EQ(A(0)(0, 0), 1.);
  ASSERT_EQ(A(0)(1, 0), 2.);
  ASSERT_EQ(A(0)(0, 1), 3.);
  ASSERT_EQ(A(0)(1, 1), 4.);
  ASSERT_EQ(A(0)(0, 2), 5.);
  ASSERT_EQ(A(0)(1, 2), 6.);
  ASSERT_EQ(A(1)(0, 0), 7.);
  ASSERT_EQ(A(1)(1, 0), 8.);
  ASSERT_EQ(A(1)(0, 1), 9.);
  ASSERT_EQ(A(1)(1, 1), 10.);
  ASSERT_EQ(A(1)(0, 2), 11.);
  ASSERT_EQ(A(1)(1, 2), 12.);
  ASSERT_EQ(A(2)(0, 0), 13.);
  ASSERT_EQ(A(2)(1, 0), 14.);
  ASSERT_EQ(A(2)(0, 1), 15.);
  ASSERT_EQ(A(2)(1, 1), 16.);
  ASSERT_EQ(A(2)(0, 2), 17.);
  ASSERT_EQ(A(2)(1, 2), 18.);
  ASSERT_EQ(A(3)(0, 0), 19.);
  ASSERT_EQ(A(3)(1, 0), 20.);
  ASSERT_EQ(A(3)(0, 1), 21.);
  ASSERT_EQ(A(3)(1, 1), 22.);
  ASSERT_EQ(A(3)(0, 2), 23.);
  ASSERT_EQ(A(3)(1, 2), 24.);

  ASSERT_EQ(A.getVal(0, 0, 0).real(), 1.);
  ASSERT_EQ(A.getVal(0, 1, 0).real(), 2.);
  ASSERT_EQ(A.getVal(0, 0, 1).real(), 3.);
  ASSERT_EQ(A.getVal(0, 1, 1).real(), 4.);
  ASSERT_EQ(A.getVal(0, 0, 2).real(), 5.);
  ASSERT_EQ(A.getVal(0, 1, 2).real(), 6.);
  ASSERT_EQ(A.getVal(1, 0, 0).real(), 7.);
  ASSERT_EQ(A.getVal(1, 1, 0).real(), 8.);
  ASSERT_EQ(A.getVal(1, 0, 1).real(), 9.);
  ASSERT_EQ(A.getVal(1, 1, 1).real(), 10.);
  ASSERT_EQ(A.getVal(1, 0, 2).real(), 11.);
  ASSERT_EQ(A.getVal(1, 1, 2).real(), 12.);
  ASSERT_EQ(A.getVal(2, 0, 0).real(), 13.);
  ASSERT_EQ(A.getVal(2, 1, 0).real(), 14.);
  ASSERT_EQ(A.getVal(2, 0, 1).real(), 15.);
  ASSERT_EQ(A.getVal(2, 1, 1).real(), 16.);
  ASSERT_EQ(A.getVal(2, 0, 2).real(), 17.);
  ASSERT_EQ(A.getVal(2, 1, 2).real(), 18.);
  ASSERT_EQ(A.getVal(3, 0, 0).real(), 19.);
  ASSERT_EQ(A.getVal(3, 1, 0).real(), 20.);
  ASSERT_EQ(A.getVal(3, 0, 1).real(), 21.);
  ASSERT_EQ(A.getVal(3, 1, 1).real(), 22.);
  ASSERT_EQ(A.getVal(3, 0, 2).real(), 23.);
  ASSERT_EQ(A.getVal(3, 1, 2).real(), 24.);
    
}

TEST(MixingParameter, ConvComplex) {
	const char* str = "<A adaptability='free' mixing_type='conv'>"
		"<ndims>3</ndims>"
		"<dim>2</dim>"
		"<dim>3</dim>"
		"<dim>4</dim>"
		"<type>complex</type>"
		"<data>1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11. 12. 13. 14. 15. 16. 17. 18. "
		"19. 20. 21. 22. 23. 24. 25. 26. 27. 28. 29. 30. 31. 32. 33. 34. 35. 36. "
		"37. 38. 39. 40. 41. 42. 43. 44. 45. 46. 47. 48.</data>"
		"</A>";
	XMLDocument xmlDoc;
	ASSERT_TRUE(xmlDoc.Parse(str) == XML_SUCCESS);
	MixingParameter A(xmlDoc.FirstChildElement("A"));

	ASSERT_FALSE(A.isInst());
	ASSERT_TRUE(A.isConv());
	ASSERT_EQ(A.rank(), 3);
	ASSERT_EQ(A.size(), 4); // nb of freq bins

  ASSERT_EQ(A(0)(0, 0), complex<double>(1, 7));
  ASSERT_EQ(A(0)(1, 0), complex<double>(2, 8));
  ASSERT_EQ(A(0)(0, 1), complex<double>(3, 9));
  ASSERT_EQ(A(0)(1, 1), complex<double>(4, 10));
  ASSERT_EQ(A(0)(0, 2), complex<double>(5, 11));
  ASSERT_EQ(A(0)(1, 2), complex<double>(6, 12));
  ASSERT_EQ(A(1)(0, 0), complex<double>(13, 19));
  ASSERT_EQ(A(1)(1, 0), complex<double>(14, 20));
  ASSERT_EQ(A(1)(0, 1), complex<double>(15, 21));
  ASSERT_EQ(A(1)(1, 1), complex<double>(16, 22));
  ASSERT_EQ(A(1)(0, 2), complex<double>(17, 23));
  ASSERT_EQ(A(1)(1, 2), complex<double>(18, 24));
  ASSERT_EQ(A(2)(0, 0), complex<double>(25, 31));
  ASSERT_EQ(A(2)(1, 0), complex<double>(26, 32));
  ASSERT_EQ(A(2)(0, 1), complex<double>(27, 33));
  ASSERT_EQ(A(2)(1, 1), complex<double>(28, 34));
  ASSERT_EQ(A(2)(0, 2), complex<double>(29, 35));
  ASSERT_EQ(A(2)(1, 2), complex<double>(30, 36));
  ASSERT_EQ(A(3)(0, 0), complex<double>(37, 43));
  ASSERT_EQ(A(3)(1, 0), complex<double>(38, 44));
  ASSERT_EQ(A(3)(0, 1), complex<double>(39, 45));
  ASSERT_EQ(A(3)(1, 1), complex<double>(40, 46));
  ASSERT_EQ(A(3)(0, 2), complex<double>(41, 47));
  ASSERT_EQ(A(3)(1, 2), complex<double>(42, 48));

  ASSERT_EQ(A.getVal(0, 0, 0), complex<double>(1, 7));
  ASSERT_EQ(A.getVal(0, 1, 0), complex<double>(2, 8));
  ASSERT_EQ(A.getVal(0, 0, 1), complex<double>(3, 9));
  ASSERT_EQ(A.getVal(0, 1, 1), complex<double>(4, 10));
  ASSERT_EQ(A.getVal(0, 0, 2), complex<double>(5, 11));
  ASSERT_EQ(A.getVal(0, 1, 2), complex<double>(6, 12));
  ASSERT_EQ(A.getVal(1, 0, 0), complex<double>(13, 19));
  ASSERT_EQ(A.getVal(1, 1, 0), complex<double>(14, 20));
  ASSERT_EQ(A.getVal(1, 0, 1), complex<double>(15, 21));
  ASSERT_EQ(A.getVal(1, 1, 1), complex<double>(16, 22));
  ASSERT_EQ(A.getVal(1, 0, 2), complex<double>(17, 23));
  ASSERT_EQ(A.getVal(1, 1, 2), complex<double>(18, 24));
  ASSERT_EQ(A.getVal(2, 0, 0), complex<double>(25, 31));
  ASSERT_EQ(A.getVal(2, 1, 0), complex<double>(26, 32));
  ASSERT_EQ(A.getVal(2, 0, 1), complex<double>(27, 33));
  ASSERT_EQ(A.getVal(2, 1, 1), complex<double>(28, 34));
  ASSERT_EQ(A.getVal(2, 0, 2), complex<double>(29, 35));
  ASSERT_EQ(A.getVal(2, 1, 2), complex<double>(30, 36));
  ASSERT_EQ(A.getVal(3, 0, 0), complex<double>(37, 43));
  ASSERT_EQ(A.getVal(3, 1, 0), complex<double>(38, 44));
  ASSERT_EQ(A.getVal(3, 0, 1), complex<double>(39, 45));
  ASSERT_EQ(A.getVal(3, 1, 1), complex<double>(40, 46));
  ASSERT_EQ(A.getVal(3, 0, 2), complex<double>(41, 47));
  ASSERT_EQ(A.getVal(3, 1, 2), complex<double>(42, 48));

}

// Tests throwing exceptions
TEST(MixingParameter, NDim3InstShouldThrow) {
  const char* str = "<A adaptability=\"free\" mixing_type=\"inst\">"
                "<ndims>3</ndims>"
                "<dim>2</dim>"
                "<dim>1</dim>"
                "<dim>1</dim>"
                "<type>real</type>"
                "<data>1 0 </data>"
                "</A>";

  XMLDocument xmlDoc;
  ASSERT_TRUE(xmlDoc.Parse(str) == XML_SUCCESS);
 ASSERT_THROW(MixingParameter A(xmlDoc.FirstChildElement("A")), runtime_error);
}

TEST(MixingParameter, ComplexInstShouldThrow) {
	const char* str = "<A adaptability=\"free\" mixing_type=\"inst\">"
                "<ndims>2</ndims>"
                "<dim>2</dim>"
                "<dim>1</dim>"
                "<type>complex</type>"
                "<data>1 0 </data>"
                "</A>";

  XMLDocument xmlDoc;
  ASSERT_TRUE(xmlDoc.Parse(str) == XML_SUCCESS);
  ASSERT_THROW(MixingParameter A(xmlDoc.FirstChildElement("A")), runtime_error);
}

TEST(MixingParameter, NDim2ConvShouldThrow) {
	const char* str = "<A adaptability=\"free\" mixing_type=\"conv\">"
                "<ndims>2</ndims>"
                "<dim>2</dim>"
                "<dim>1</dim>"
                "<type>complex</type>"
                "<data>1 0 </data>"
                "</A>";

	XMLDocument xmlDoc;
	ASSERT_TRUE(xmlDoc.Parse(str) == XML_SUCCESS);
	ASSERT_THROW(MixingParameter A(xmlDoc.FirstChildElement("A")), runtime_error);
}