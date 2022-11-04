//===========================================================================
/*!
 *
 *
 * \brief       General functions for Tree modeling.
 *
 *
 *
 * \author      K. N. Hansen, J. Kremer, J. Wrigley
 * \date        2011-2016
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
 *
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 *
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================
#ifndef SHARK_ALGORITHMS_TRAINERS_CARTCOMMON_H
#define SHARK_ALGORITHMS_TRAINERS_CARTCOMMON_H

#include <shark/Models/Trees/General.h>
#include <shark/Data/DataView.h>
#include <vector>
#include <unordered_map>
#include <utility>
namespace shark {
namespace detail {
namespace cart {
/// ClassVector
using ClassVector = UIntVector;


struct Attribute {
	double value;
	std::size_t id;

	inline bool operator<(Attribute const &v) const {
		return value < v.value;
	}
};

/**
 * An Index of a dataset providing fast sorted-order lookup of each attribute.
 */
class SortedIndex {
/// attribute table
	using AttributeTable = std::vector<Attribute>;
/// collection of attribute tables
	using AttributeTables = std::vector<AttributeTable>;

	std::size_t m_noElements, m_noInputDimensions;
	AttributeTables m_tables;

	explicit SortedIndex(AttributeTables &&tables)
			: m_noElements(tables[0].size()),
			  m_noInputDimensions(tables.size()),
			  m_tables(std::move(tables))
	{}
public:
/** Creates an index of the dataset
 *  A dataset consisting of m input variables has m sorted attribute tables.
 *  [attribute | rid ]
 */
	template<class Dataset>
	explicit SortedIndex(DataView<Dataset const> const &elements)
			: m_noElements{elements.size()},
			  m_noInputDimensions{elements[0].input.size()},
			  m_tables(m_noInputDimensions)
	{
		std::size_t n_elements = m_noElements;
		//Each entry in the outer vector is an attribute table
		//For each column
		for (std::size_t j = 0; j < m_noInputDimensions; j++) {
			auto &table = m_tables[j];
			table.reserve(n_elements);

			//For each row, store Attribute value, class and rowId
			for (std::size_t i = 0; i < n_elements; i++) {
				table.push_back(Attribute{elements[i].input[j], i});
			}
			std::sort(table.begin(), table.end());
		}
	}

/**
 * Returns two Indices: left and right
 * Calculated from splitting tables at (index, valIndex)
 */
	std::pair<SortedIndex, SortedIndex> split(std::size_t index, std::size_t valIndex) {
		//Build a hash table for fast lookup
		std::unordered_map<std::size_t, bool> hash;
		for(std::size_t i = 0, s = m_tables[index].size(); i<s; ++i) {
			hash[m_tables[index][i].id] = (i<=valIndex);
		}

		AttributeTables RAttributeTables, LAttributeTables;
		for(auto && table : m_tables) {
			auto begin = table.begin(), end = table.end();
			auto middle = std::stable_partition(begin,end,[&hash](Attribute const& entry){
				return hash[entry.id];
			});
			RAttributeTables.emplace_back(AttributeTable{middle,end});
			table.resize(std::distance(begin,middle));
			LAttributeTables.emplace_back(std::move(table));
		}
		m_tables.clear(); m_tables.shrink_to_fit();
		return std::make_pair(SortedIndex{std::move(LAttributeTables)},
							  SortedIndex{std::move(RAttributeTables)});
	}

	std::size_t noTables() const { return m_noInputDimensions; }
	std::size_t noRows() const { return m_noElements; }
	std::size_t size() const { return noTables(); }
	inline AttributeTable const& operator[](std::size_t i) const {
		return m_tables[i];
	}
};

/// Generate a histogram from the count vector.
inline RealVector hist(ClassVector const& countVector) {
	return countVector/double(sum(countVector));
}

using ImpurityMeasure = double (*)(ClassVector const& countVector, std::size_t n);

/// Calculate the Gini impurity of the countVector
double gini(ClassVector const& countVector, std::size_t n);
double misclassificationError(ClassVector const& countVector, std::size_t n);
double crossEntropy(ClassVector const& countVector, std::size_t n);

/// Create a count vector as used in the classification case.
ClassVector createCountVector(DataView<ClassificationDataset const> const& elements, std::size_t labelCardinality);

}}} // namespace shark::detail::cart

#endif //SHARK_ALGORITHMS_TRAINERS_CARTCOMMON_H
