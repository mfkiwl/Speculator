#include "IrSequencer.hpp"

//============================================================
ButtonGrid::ButtonGrid(HashMap<String, IRState>& sourceIRStates):
storedIrStates(sourceIRStates)
{
    for (int i = 0; i < numElements; ++i)
    {
        auto newToggleButton = new ToggleButton ();
        newToggleButton->setLookAndFeel(&lookAndFeel);
        newToggleButton->addListener(this);
        newToggleButton->getToggleStateValue().addListener(this);
        auto name = "ButtonCell-" + String(i);
        newToggleButton->setName(name);
        addAndMakeVisible(newToggleButton);
        buttonCells.add(newToggleButton);
    }
    
    for (int row = 0; row < numRows; row++) {
        auto newIRComboBox = new ComboBox();
        newIRComboBox->addListener(this);
        newIRComboBox->setName("IRComboBox_" + String(row));
        newIRComboBox->getSelectedIdAsValue().addListener(this);
        addAndMakeVisible(newIRComboBox);
        newIRComboBox->setTextWhenNoChoicesAvailable("ir -" + String(row+1));
        irComboBoxes.add(newIRComboBox);
    }

    Font buttonFont("courier",12,0);
    addAndMakeVisible(storeSequenceButton);
    storeSequenceButton.setButtonText("add seq");
    storeSequenceButton.addListener(this);
    
    addAndMakeVisible(overwriteSequenceButton);
    overwriteSequenceButton.setButtonText("overwrite seq");
    overwriteSequenceButton.addListener(this);
    
    addAndMakeVisible(removeSequenceButton);
    removeSequenceButton.setButtonText("remove seq");
    removeSequenceButton.addListener(this);
    
    addAndMakeVisible(setSequenceButton);
    setSequenceButton.setButtonText("set seqs");
    setSequenceButton.addListener(this);
    
    addAndMakeVisible(clearSequencesButton);
    clearSequencesButton.setButtonText("clear all");
    clearSequencesButton.addListener(this);
    
    addAndMakeVisible(clearCurrentSequenceButton);
    clearCurrentSequenceButton.setButtonText("clear seq");
    clearCurrentSequenceButton.addListener(this);
    
    addAndMakeVisible(sequenceNameLabel);
    sequenceNameLabel.setText("default", dontSendNotification);
    sequenceNameLabel.setEditable(true);
    sequenceNameLabel.setFont(buttonFont);
    sequenceNameLabel.setJustificationType(Justification::centred);
    
    addAndMakeVisible(sequencesComboBox);
    sequencesComboBox.setText(" --- ");
    sequencesComboBox.addListener(this);
    
    setParametersToReferToValueTreeProperties();
    currentSequenceState.setProperty("name", "default", nullptr);
    currentSequenceState.setProperty("uid", "-1", nullptr);
    storeCurrentSequence();
    

}
//============================================================
void ButtonGrid::paint(juce::Graphics &g)
{
    for (const auto& buttonCell : buttonCells)
        g.drawRect(buttonCell->getBounds());
    
    g.setColour(Colours::lightgrey);
    g.fillRoundedRectangle(sequenceNameLabel.getBounds().toFloat(), 3.f);
}
 //============================================================}
void ButtonGrid::paintOverChildren (juce::Graphics &g)
{
    int columnWidth = (float)gridBounds.getWidth()/(float)numColumns;
    
    auto columnBounds = gridBounds;
    for (int column = 0; column < numColumns; column++)
    {
        auto currentColumnBounds = columnBounds.removeFromLeft(columnWidth);
        g.setColour(Colours::darkgrey.withAlpha((float)0.5f));
        if (!columnHasActiveCell(column))
            g.fillRect(currentColumnBounds);
    }
}
//============================================================
void ButtonGrid::resized()
{
    auto r = getLocalBounds();
    
    auto buttonColumnBounds = r.removeFromLeft(100);
    auto comboBoxColumn = r.removeFromLeft(100);
    
    gridBounds = r;
    
    const int columnWidth = (float)gridBounds.getWidth()/(float)numColumns;
    const int rowHeight = (float)gridBounds.getHeight()/(float)numRows;
    
    sequenceNameLabel.setBounds(buttonColumnBounds.removeFromTop(25));
    auto storeLoadSequenceBounds = buttonColumnBounds.removeFromTop(25);
    storeSequenceButton.setBounds(storeLoadSequenceBounds.removeFromLeft(50));
    overwriteSequenceButton.setBounds(storeLoadSequenceBounds);
    setSequenceButton.setBounds(buttonColumnBounds.removeFromTop(25));
    removeSequenceButton.setBounds(buttonColumnBounds.removeFromTop(25));
    sequencesComboBox.setBounds(buttonColumnBounds.removeFromTop(25));
    clearCurrentSequenceButton.setBounds(buttonColumnBounds.removeFromTop(25));
    clearSequencesButton.setBounds(buttonColumnBounds.removeFromTop(25));
    
    for (int row = 0; row < numRows; row++) {
        irComboBoxes[row]->setBounds(comboBoxColumn.removeFromTop(rowHeight).reduced(2, 10));
    }
    
    for (int column = 0; column < numColumns; column++) {
        
        auto columnBounds = r.removeFromLeft(columnWidth);
        
        for (int row = 0; row < numRows; row++) {
            
            int n = column * numRows + row;
//            std::cout << "n: " << n << "row: " << row << " column: " << column << std::endl;
            buttonCells[n]->setBounds(columnBounds.removeFromTop(rowHeight));
        }
    }
}
//============================================================
void ButtonGrid::buttonClicked(Button* clickedButton)
{
    for (int column = 0; column < numColumns; column++) {
        for (int row = 0; row < numRows; row++) {
            
            int n = column * numRows + row;

            if (clickedButton == buttonCells[n]) {
//                std::cout << "n: " << n << "row: " << row << " column: " << column << std::endl;
                bool tempButtonState = clickedButton->getToggleState();
                deselectAllCellsInColumn(column);
                buttonCells[n]->setToggleState(tempButtonState, dontSendNotification);
                
                break;
            }
        }
    }
    
    if (clickedButton == &storeSequenceButton) storeCurrentSequence();
    if (clickedButton == &overwriteSequenceButton) overwriteCurrentSequence();
    if (clickedButton == &removeSequenceButton) removeCurrentSequence();
    if (clickedButton == &setSequenceButton) setCurrentSequence();
    if (clickedButton == &clearCurrentSequenceButton) clearCurrentSequence();
    if (clickedButton == &clearSequencesButton) clearAllSequences();
        
}
//============================================================
void ButtonGrid::comboBoxChanged(juce::ComboBox *changedComboBox)
{
    if (changedComboBox == &sequencesComboBox) {
        String comboBoxText = changedComboBox->getText();
        
        if (irSequences.contains(comboBoxText))
            waitingSequenceState = sequenceStates[comboBoxText];
    }
}
//============================================================
void ButtonGrid::valueChanged(juce::Value &changedValue)
{
    for (int column = 0; column < numColumns; column++) {
        for (int row = 0; row < numRows; row++) {
            int n = column * numRows + row;
            if (changedValue == buttonCells[n]->getToggleStateValue()) {
                
                if (irSequences.contains(getCurrentStateName())) {
                    auto& irElement = irSequences[getCurrentStateName()]
                    .getReference(column);
                    
                    int selectedRowIndex = getActiveRowInColumn(column);

                    irElement.isEnabled = (bool)changedValue.getValue();
                    
                    int irClipIndex = 0;
                    if (irElement.isEnabled) {
//                        std::cout << getCurrentStateName() << std::endl;
                        irClipIndex = irComboBoxes[selectedRowIndex]->getSelectedId();
                        
                        irElement.irClipName = irComboBoxes[selectedRowIndex]->getItemText(irClipIndex);
                    };
                
                    irElement.irClipindex = irClipIndex;
                }
                
                break;
            }
        }
    }
    
}
//============================================================
const bool ButtonGrid::getToggleStateAt(int n)
{
    return buttonCells[n]->getToggleState();
}
//============================================================
const bool ButtonGrid::getToggleStateAt(int row, int column)
{
    return getToggleStateAt(row * numRows + column);
}
//============================================================
void ButtonGrid::addItemToIRComboBoxes(juce::String itemName, int itemID)
{
    for (auto irComboBox : irComboBoxes)
        irComboBox->addItem(itemName, itemID);
}
//============================================================
void ButtonGrid::clearIRComboBoxes()
{
    for (auto irCombobox : irComboBoxes)
        irCombobox->clear();
}
//============================================================
void ButtonGrid::storeCurrentSequence()
{
    String newSequenceName = sequenceNameLabel.getText();
    int newUID = 1;
    for (int i = 0; i < sequencesComboBox.getNumItems(); i++) {
        if (sequencesComboBox.getItemId(i) == newUID) {
            newUID++;
        }
    }

    if (!sequenceStates.contains(newSequenceName))
    {
        std::cout << "creating new - uid: " << newUID << " name " << newSequenceName << std::endl;
        ValueTree newSequenceState;
        newSequenceState = currentSequenceState.createCopy();
        newSequenceState.setProperty("uid", newUID, nullptr);
        newSequenceState.setProperty("name", newSequenceName, nullptr);
        
        sequenceStates.set(newSequenceName, std::move(newSequenceState));
        sequencesComboBox.addItem(newSequenceName, newUID);

        irSequences.set(newSequenceName,std::move(generateIRSequenceFromCurrentState()));
    }
}
void ButtonGrid::overwriteCurrentSequence()
{
    String newSequenceName = sequenceNameLabel.getText();
    bool nameAlreadyUsed = false;
    int newUID = 1;
    for (const auto& sequenceState : sequenceStates)
    {
        nameAlreadyUsed = (sequenceState.getProperty("name").toString() == newSequenceName);
        if ((int)sequenceState.getProperty("uid") == newUID) newUID++;
    }
  
        std::cout << "creating new - uid: " << newUID << " name " << newSequenceName << std::endl;
    ValueTree newSequenceState = std::move(currentSequenceState.createCopy());
        newSequenceState.setProperty("uid", newUID, nullptr);
        newSequenceState.setProperty("name", newSequenceName, nullptr);
        
        sequenceStates.set(newSequenceName, newSequenceState);
        sequencesComboBox.addItem(newSequenceName, newUID);
        
        irSequences.set(newSequenceName, std::move(generateIRSequenceFromCurrentState()));
    
}
void ButtonGrid::setCurrentSequence()
{
//    String selectedItemName = sequencesComboBox.getText();
//    currentSequenceState.copyPropertiesFrom(sequenceStates[selectedItemName], nullptr);
    
    currentSequenceState.copyPropertiesFrom(waitingSequenceState, nullptr);
}
void ButtonGrid::removeCurrentSequence()
{
//    String
    if (sequenceStates.contains(getCurrentStateName()))
    {
        irSequences.remove(getCurrentStateName());
        sequenceStates.remove(getCurrentStateName());
        sequencesComboBox.clear();
        
        for (const auto& sequenceState : sequenceStates) {
            sequencesComboBox.addItem(sequenceState.getProperty("name").toString(),
                                      (int)sequenceState.getProperty("uid"));
        }
    }

}
void ButtonGrid::clearCurrentSequence()
{
    performFunctionOnCells([] (ToggleButton* buttonCell, int row, int){
        buttonCell->setToggleState(false, dontSendNotification);
    });
}
void ButtonGrid::clearAllSequences()
{}
//============================================================
void ButtonGrid::setParametersToReferToValueTreeProperties()
{
    for (auto buttonCell: buttonCells)
    {
        buttonCell->getToggleStateValue().referTo(
                                                  currentSequenceState.getPropertyAsValue(getButtonStatePropertyName(buttonCell),nullptr));
    }
    
    for (auto irComboBox : irComboBoxes)
    {
        for (int itemIndex = 0; itemIndex < irComboBox->getNumItems(); itemIndex++)
        {
            currentSequenceState.setProperty( getComboBoxIDPropertyName(irComboBox,
                                                         itemIndex),
                              irComboBox->getSelectedId(),
                              nullptr);
            
            currentSequenceState.setProperty(
                              getComboBoxItemTextPropertyName(irComboBox,
                                                              itemIndex),
                              irComboBox->getItemText(itemIndex),
                              nullptr);
        }
    }
    
}
String ButtonGrid::getCurrentStateName()
{
    return currentSequenceState.getProperty("name").toString();
}
//============================================================
IRSequence ButtonGrid::generateIRSequenceFromCurrentState()
{
    IRSequence newIrSequence;
    newIrSequence.ensureStorageAllocated(numColumns);
    
    for (int columnIndex = 0; columnIndex < numColumns; columnIndex++)
    {
        int selectedRowIndex = getActiveRowInColumn(columnIndex);
        
        bool isEnabled = (selectedRowIndex > -1);
        int irClipIndex = 0;
        String clipName;
        if (isEnabled) {
            irClipIndex = irComboBoxes[selectedRowIndex]->getSelectedId();
            clipName = irComboBoxes[selectedRowIndex]->getItemText(irClipIndex);
        };
        
        newIrSequence.add(std::forward<IRSequenceElement>({ columnIndex, irClipIndex, clipName, isEnabled }));
    }
    
    return newIrSequence;
}
//============================================================
template<class Function>
void ButtonGrid::performFunctionOnColumn(int columnIndex,
                                         Function functionToPerform)
{
    for (int row = 0; row < numRows; row++)
    {
        functionToPerform(buttonCells[columnIndex * numRows + row]);
    }
}
template<class Function>
void ButtonGrid::performFunctionOnCells(Function functionToPerform)
{
    for (int column = 0; column < numColumns; column++) {
        for (int row = 0; row < numRows; row++) {
            int cellIndex = column * numRows + row;
            functionToPerform(buttonCells[cellIndex], column, row);
        }
    }
}
bool ButtonGrid::columnHasActiveCell(int columnIndexToCheck)
{
    bool hasActiveCell = false;
    for (int rowIndex = 0; rowIndex < numRows; rowIndex++)
    {
        int index = columnIndexToCheck * numRows + rowIndex;
        if (buttonCells[index]->getToggleState()) {
            hasActiveCell = true;
            break;
        }
    }
    return hasActiveCell;
}

int ButtonGrid::getActiveRowInColumn(int columnIndex)
{
    for (int rowIndex = 0; rowIndex < numRows; rowIndex++)
    {
        int index = columnIndex * numRows + rowIndex;
        if (buttonCells[index]->getToggleState())
            return rowIndex;
    }
    return -1;
}
//============================================================
void ButtonGrid::deselectAllCellsInColumn(int columnIndex)
{
    performFunctionOnColumn(columnIndex, [](ToggleButton* button){
        button->setToggleState(false, dontSendNotification);
    });
}






