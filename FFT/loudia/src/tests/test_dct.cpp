/*                                                         
** Copyright (C) 2008, 2009 Ricard Marxer <email@ricardmarxer.com>
**                                                                  
** This program is free software; you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation; either version 3 of the License, or   
** (at your option) any later version.                                 
**                                                                     
** This program is distributed in the hope that it will be useful,     
** but WITHOUT ANY WARRANTY; without even the implied warranty of      
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the       
** GNU General Public License for more details.                        
**                                                                     
** You should have received a copy of the GNU General Public License   
** along with this program; if not, write to the Free Software         
** Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307, USA.
*/                                                                          

#include "DCT.h"

#include "Typedefs.h"
#include "Debug.h"

#include <Eigen/Core>
#include <iostream>

using namespace std;

int main() {
  int inputLength = 34;
  int dctLength = 12;

  MatrixXR in = MatrixXR::Constant(1, inputLength, 1.0);
  
  DCT dct(inputLength, dctLength);
  dct.setup();

  MatrixXR result(1, dctLength);
  
  for (int i=0; i<2; i++) {   
    dct.process(in, &result);
    cout << "result:" << result << endl;
  }

  return 0;
}

