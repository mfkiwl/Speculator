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


#ifndef __CREATEPROCESSDIALOG_H__
#define __CREATEPROCESSDIALOG_H__


#include "ui_CreateProcessDialog.h"
#include "ThreadedDialog.h"

#include <blissart/BasicTask.h>

#include <Poco/Mutex.h>

#include <set>


namespace blissart {


/**
 * \addtogroup browser
 * @{
 */

/**
 * Represents a dialog that can be used to load and separate audio files
 * into the database.
 */
class CreateProcessDialog : public ThreadedDialog
{
    Q_OBJECT


public:
    CreateProcessDialog(QWidget *parent = 0);


public slots:
    /**
     * Handles the dialog's accept signal and performs the process creation.
     */
    virtual void accept();


protected:
    /**
     * Shows a dialog box displaying the names of those files that were not
     * processed.
     */
    void showFailedFileNames();


    /**
     * Allows for logging the filenames of corresponding tasks in case they were
     * cancelled or have failed.
     */
    virtual void taskAboutToBeRemoved(BasicTaskPtr task);


protected slots:
    void on_cbProcessing_currentIndexChanged(int index);


private:
    Ui::CreateProcessDialog        _ui;

    Poco::FastMutex                _genMutex;
    std::vector<std::string>       _failedFileNames;
};


/**
 * @}
 */
    

} // namespace blissart


#endif
