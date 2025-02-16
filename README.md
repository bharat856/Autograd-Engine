This Jupyter Notebook implements an automatic differentiation engine similar to PyTorch’s autograd. The core functionality revolves around computing gradients (derivatives) automatically, which is essential for machine learning and deep learning.
Key Features
	1.	Custom Computational Graph Implementation:
	•	Defines a Value class that represents scalar values.
	•	Supports mathematical operations (+, -, *, /, **, tanh, exp).
	•	Tracks dependencies (previous operations) to build a computation graph.
	2.	Gradient Calculation with Backpropagation:
	•	Implements automatic differentiation using the backward() method.
	•	Uses reverse-mode differentiation (backpropagation).
	•	Ensures correct accumulation of gradients (+= instead of =) to handle repeated variables.
	3.	Visualization of the Computation Graph:
	•	Uses Graphviz to create a visual representation of the computational graph.
	•	Helps in understanding how operations contribute to the final output.
	4.	Basic Mathematical Functions for Testing:
	•	Example function:  f(x) = 3x^2 - 4x + 5 
	•	Generates sample inputs using NumPy.
	•	Computes derivatives of functions using the autograd engine.
How It Works
	•	When an operation (like addition, multiplication) is performed between Value objects, a new Value instance is created with a reference to its parent nodes.
	•	Each operation defines a backward function to compute its contribution to the gradient.
	•	Calling .backward() on a final output triggers reverse-mode differentiation, computing gradients for all intermediate values.
