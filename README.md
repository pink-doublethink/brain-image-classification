----
### What is the project about?
----
- This repository is dedicated to classifying MRI head images to detect various types of tumors, including "Meningioma," "Glioma," "Pituitary," and "No tumor." The dataset is sourced from Kaggle's Brain Tumor MRI Dataset, comprising four class-specific directories.
- The model involves image preprocessing techniques like flipping, rescaling, normalization, zooming, and rotation, exclusively applied to the training dataset. Training utilizes pretrained Resnet50 and EfficientNet models, with the final layer adapted to match the number of classes. Three model versions are available, two trained with Torch (using EfficientNet and Resnet50) and one with TensorFlow without pretrained layers.
----
### Contributing
----
- Contributions to this project are welcome. If you find a bug or have a feature request, please open an issue on the project's GitHub page. If you would like to contribute code, please fork the repository and create a pull request with your changes.