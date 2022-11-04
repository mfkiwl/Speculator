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


#ifndef __FILESSELECTIONWIDGET_H__
#define __FILESSELECTIONWIDGET_H__


#include "ui_FilesSelectionWidget.h"
#include <QStringList>


namespace blissart {


/**
 * \addtogroup browser
 * @{
 */

/**
 * Implements a widget that edits a file list. 
 * Files can be added from various directories via a dialog.
 */
class FilesSelectionWidget : public QWidget
{
    Q_OBJECT
    
public:
    /**
     * Constructs a new instance of FilesSelectionWidget.
     */
    FilesSelectionWidget(QWidget *parent);
    
    
    /**
     * Returns a list of all contained files.
     */
    QStringList fileNames() const;
    
    
protected slots:
    /**
     * Handles the addition of one or more files.
     */
    void on_pbAddFiles_clicked();
    
    
    /**
     * Removes the selected files.
     */
    void on_pbRemoveSelected_clicked();
    

private:
    Ui::FilesSelectionWidget _ui;
};


/**
 * @}
 */
    

} // namespace blissart


#endif
