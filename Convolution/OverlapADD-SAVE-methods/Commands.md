# MATLAB Commands and data structures used
1. zeros()-Create array of all zeros
2. cconv()-Modulo-n circular convolution
3. cell2mat()- Convert cell array to ordinary array of the underlying data typeA = cell2mat(C) converts a cell array into an ordinary array. The elements of the cell array must all contain the same data type, and the resulting array is of that data type.The contents of C must support concatenation into an N-dimensional rectangle. 
4. cell- A cell array is a data type with indexed data containers called cells, where each cell can contain any type of data. Cell arrays commonly contain either lists of text, combinations of text and numbers, or numeric arrays of different sizes. Refer to sets of cells by enclosing indices in smooth parentheses, (). Access the contents of cells by indexing with curly braces, {}.
5. input()- _Request user input_.input(prompt) displays the text in prompt and waits for the user to input a value and press the Return key. The user can enter expressions, like pi/4 or rand(3), and can use variables in the workspace.

    If the user presses the Return key without entering anything, then input returns an empty matrix.

    If the user enters an invalid expression at the prompt, then MATLAB® displays the relevant error message, and then redisplays the prompt.
6. disp()-Display value of variable.disp(X) displays the value of variable X without printing the variable name. Another way to display a variable is to type its name, which displays a leading “X =” before the value.If a variable contains an empty array, disp returns without displaying anything.
