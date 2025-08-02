# Neural Networks
- What is a Neural Network ?

A neural network is a type of machine learning model inspired by the structure of the human brain. Just like your brain has neurons that are connected and process information, a neural network is made up of artificial neurons (also called nodes or units) that work together to learn patterns from data. 
## Structure of Neural Network:
1. Input Layer: This layer receives the input features like pixels of an image, values from a spreadsheet, etc.
2. Hidden Layer: These layers do the actual computation it is present between the input and the output layers. There can be multiple hidden layers.
3. Output Layer: The layer which gives us the final output.
## Supervised Learning with Neural Networks
Supervised learning is a type of machine learning where the model learns from labeled data — meaning, the input comes with the correct answer.

Think of it like teaching a child with flashcards:

You show a picture of a cat (input) and say “This is a cat” (label).

Do this enough times, and the child (or model) starts recognizing cats on its own!
- Structured Data: Data that is organized in a tabular format with defined features.Each column has clear meaning and can be directly used for prediction tasks.

For example, in the house price prediction we can use features like number of bedrooms, size, zip code etc.

- Unstructured Data: Data that isn’t organized in a predefined format and needs to be processed to extract useful information.

For example: Images, audio files, and text documents are unstructured and need models like CNNs or RNNs to interpret them.

## Why is Deep Learning taking off ?
1. More Data
2. Better Hardware (GPUs) -> Graphic processing unit is a special type of computer chip designed to handle lots of small calculation at same time.
3. Scalability
4. Open Source Tools (TensorFlow, PyTorch)

## Types of Neural Networks
1. Feedforward Neural Networks(FNN): Data flows in one direction from input to output, with no cycles.
2. Convolutional Neural Networks(CNN): Specialized for processing image data using filters to detect features like edges and textures.
3. Recurrent Neural Networks(RNN): Designed for sequential data, where each output depends on previous steps.
4. Long Short-Term Memory(LSTM): An improved RNN that captures long-term dependencies more effectively in sequences.
5. Transformer Networks: Uses attention mechanisms instead of recurrence to handle long-range dependencies, especially in text data.