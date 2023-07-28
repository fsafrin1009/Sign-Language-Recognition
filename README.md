Sign Language Recognition Project

This project is a complex sign language recognition system that can recognize hand signs representing not only numbers but also letters (A to Z) and different emotions. The project is implemented using TensorFlow, Python, and Jupyter Notebook.

Key Features
- Sign language recognition for numbers (0 to 25), letters (A to Z), and emotions (e.g., happy, sad, surprised, etc.).
- Convolutional Neural Network (CNN) architecture for image classification.
- Data preprocessing and normalization to enhance model performance.
- Evaluation of the model's accuracy using a test dataset.
- Random testing to visualize the model's predictions on random test images.

Getting Started
1. Clone the repository to your local machine.
2. Install the required dependencies (TensorFlow, NumPy, matplotlib, etc.).
3. Open the Jupyter Notebook (`sign_language_recognition.ipynb`) to access the code and data.
4. Execute the code cells to train and evaluate the model.
5. Observe the training history and test accuracy.

Dataset
The project utilizes the MNIST dataset, which contains images of hand signs representing numbers. Additionally, a custom dataset is used to include hand signs representing letters (A to Z) and emotions.

Model Architecture
The CNN architecture consists of multiple convolutional layers, max-pooling layers, and fully connected layers to extract features and classify the hand signs.

Training
The model is trained on the combined dataset containing numbers, letters, and emotions. It is trained for 20 epochs with a batch size of 32.

Evaluation
The model's accuracy is evaluated using a separate test dataset to assess its performance on unseen data.

Random Testing
Random test images are selected to visually inspect the model's predictions on unseen examples.

Results
The trained model achieves an accuracy of approximately 98.55% on the test dataset, demonstrating its capability to recognize a diverse set of hand signs effectively.

Feel free to explore the project and customize it further to include additional sign categories or enhance the model's architecture for even better performance!

---

