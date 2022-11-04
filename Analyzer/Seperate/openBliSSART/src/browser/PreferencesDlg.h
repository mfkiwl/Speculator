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


#ifndef __PREFERENCESDLG_H__
#define __PREFERENCESDLG_H__


#include "ui_PreferencesDlg.h"
#include <QMap>


namespace blissart {


/**
 * \addtogroup browser
 * @{
 */

/**
 * A dialog that allows the user to change the preferences for the browser GUI.
 */
class PreferencesDlg : public QDialog
{
    Q_OBJECT
    
public:
    /**
     * Constructs a new intance of PreferencesDlg.
     */
    PreferencesDlg(QWidget *parent = 0);
    
    
public slots:
    /**
     * Handles the dialog's accept signal and stores the preferences.
     */
    virtual void accept();
    
    
protected:
    /**
     * Sets up a mapping from UI elements to configuration items. Only those
     * UI elements that are relevant for the configuration are taken into
     * account.
     */
    void setupConfigMap();
    
    
    /**
     * Walks over the mapping from UI elements to configuration items and
     * stores the respective data.
     */
    void setConfig();


    /**
     * Walks over the mapping from UI elements to configuration items and
     * restores the respective data.
     */
    void getConfig();

    
    /**
     * A mapping from UI elements to configuration items.
     */
    QMap<QWidget *, const char *> _configMap;
    

private:
    Ui::PreferencesDlg            _ui;
};


/**
 * @}
 */
    

} // namespace blissart


#endif
