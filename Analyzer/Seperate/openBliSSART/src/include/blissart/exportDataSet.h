//
// This file is part of openBliSSART.
//
// Copyright (c) 2007-2009, Alexander Lehmann <lehmanna@in.tum.de>
//                          Felix Weninger <felix@weninger.de>
//                          Bjoern Schuller <schuller@tum.de>
//
// Institute for Human-Machine Communication
// Technische Universitaet Muenchen (TUM), D-80333 Munich, Germany
//
// openBliSSART is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 2 of the License, or (at your option) any later
// version.
//
// openBliSSART is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along with
// openBliSSART.  If not, see <http://www.gnu.org/licenses/>.
//


#ifndef __BLISSART_EXPORTDATASET_H__
#define __BLISSART_EXPORTDATASET_H__


#include <common.h>
#include <string>
#include <blissart/DataSet.h>


namespace blissart {


/**
 * \addtogroup framework
 * @{
 */

/**
 * Exports a DataSet object in the ARFF format.
 *
 * @param   dataSet     The DataSet to export
 * @param   fileName    Export file name
 * @param   name        Name of the data set
 *                      (used as the @RELATION attribute)
 * @param   description Description of the data set
 *                      (inserted as a comment)
 */
bool LibFramework_API exportDataSet(const DataSet& dataSet,
                                    const std::string& fileName,
                                    const std::string& name,
                                    const std::string& description);


/**
 * @}
 */


} // namespace blissart


#endif // __BLISSART_EXPORTDATASET_H__
