# julia-image-classification-svm
The code loads dog and cat images, extracts HOG descriptors, labels them, splits the data into training and test sets, trains an SVM model, and predicts a test image.

Let's analyze the code block step by step:

### Code Analysis

#### 1. Importing Necessary Packages
```julia
using ImageFeatures, Images
using Random
using ImageDraw, ImageView
using LIBSVM
```
- **ImageFeatures**: For creating image descriptors.
- **Images**: For image loading and processing.
- **Random**: For random operations, like shuffling.
- **ImageDraw** and **ImageView**: Likely used for drawing and viewing images.
- **LIBSVM**: For training and using an SVM model.

#### 2. Defining Paths for Dog and Cat Image Directories
```julia
dog = "DATASETS/archive/PetImages/Dog"
cat = "DATASETS/archive/PetImages/Cat"
```

#### 3. Counting the Number of Images
```julia
n_dogs = length(readdir(dog)[1:8000])
n_cats = length(readdir(cat)[1:8000])
```
- **readdir(dog)[1:8000]**: Reads up to 8000 filenames from the dog directory.
- **length**: Counts the number of files read.

#### 4. Defining the Dataset Size and Initializing Arrays
```julia
total = n_cats + n_dogs
data = Array{Float64}(undef, 1764, total)
labels = Array{Int}(undef, total)
```
- **total**: Total number of images.
- **data**: A 2D array to store the HOG descriptors (1764 features per image).
- **labels**: An array to store labels (1 for dog, 0 for cat).

#### 5. Creating the Dataset
```julia
index_num = 1
for i in readdir(dog)[1:8000]
    img = load("DATASETS/archive/PetImages/Dog/$i")
    data[:, index_num] = create_descriptor(img, HOG())
    labels[index_num] = 1
    global index_num += 1
end

for i in readdir(cat)[1:8000]
    img = load("DATASETS/archive/PetImages/Cat/$i")
    data[:, index_num] = create_descriptor(img, HOG())
    labels[index_num] = 0
    global index_num += 1
end
```
- **index_num**: Tracks the current index in the dataset.
- **readdir**: Reads filenames from the dog and cat directories.
- **load**: Loads an image.
- **create_descriptor(img, HOG())**: Creates a HOG (Histogram of Oriented Gradients) descriptor for the image.
- **data[:, index_num] = create_descriptor(img, HOG())**: Stores the descriptor in the data array.
- **labels[index_num] = 1**: Sets the label to 1 for dogs.
- **labels[index_num] = 0**: Sets the label to 0 for cats.

#### 6. Splitting the Dataset into Training and Test Sets
```julia
random_perm = randperm(total)
train_id = random_perm[1:15000]
test_id = random_perm[15001:end]
```
- **randperm(total)**: Generates a random permutation of indices from 1 to `total`.
- **train_id**: First 15000 indices for training.
- **test_id**: Remaining indices for testing.

#### 7. Training the SVM Model
```julia
model = svmtrain(data[:, train_id], labels[train_id])
```
- **svmtrain**: Trains the SVM model using the training data and labels.

#### 8. Loading and Preprocessing a Test Image
```julia
my_test_img = "DATASETS/archive/PetImages/test/391.jpg"
my_test_img = load(my_test_img)
descriptor = Array{Float64}(undef, 1764, 1)
descriptor[:, 1] = create_descriptor(my_test_img, HOG())
```
- **load**: Loads the test image.
- **descriptor**: Array to store the descriptor for the test image.
- **create_descriptor(my_test_img, HOG())**: Creates a HOG descriptor for the test image.

#### 9. Predicting the Label of the Test Image
```julia
predicted_label, _ = svmpredict(model, descriptor)
print("This Photo belongs to the class: $(predicted_label[1])")
```
- **svmpredict**: Predicts the label of the test image using the trained SVM model.
- **predicted_label**: The predicted label for the test image.
- **print**: Prints the predicted class.

### Summary
This code trains an SVM classifier to differentiate between images of dogs and cats using HOG descriptors. It then predicts the class of a new test image. Here are the key steps:
1. Load and preprocess the image data.
2. Create HOG descriptors for each image.
3. Label the data (1 for dogs, 0 for cats).
4. Split the dataset into training and test sets.
5. Train an SVM model on the training data.
6. Predict the label of a new image using the trained model.

