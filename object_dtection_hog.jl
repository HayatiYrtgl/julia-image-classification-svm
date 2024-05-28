using ImageFeatures, Images
using Random
using ImageDraw, ImageView
# dog_examps and cat examps
dog = "DATASETS/archive/PetImages/Dog"
cat = "DATASETS/archive/PetImages/Cat"

# find length
n_dogs = length(readdir(dog)[1:8000])
n_cats = length(readdir(cat)[1:8000])

# total examps
total = n_cats + n_dogs

# data and label
data = Array{Float64}(undef, 1764, total)
labels = Array{Int}(undef, total)
# class 1 dog, class 2 cat
# create DATASET
index_num = 1
for (i) in readdir(dog)[1:8000]
    img = load("DATASETS/archive/PetImages/Dog/$i")
    data[:, index_num] = create_descriptor(img, HOG())
    labels[index_num] = 1
    global index_num += 1
end
println(index_num)
for (i) in readdir(cat)[1:8000]
    img = load("DATASETS/archive/PetImages/Cat/$i")
    data[:, index_num] = create_descriptor(img, HOG())
    labels[index_num] = 0
    global index_num += 1
end

using LIBSVM
# split dataset into train and test 
random_perm = randperm(total)
train_id = random_perm[1:15000]
test_id = random_perm[15001:end]

# model
model = svmtrain(data[:, train_id], labels[train_id])

# load img # preprocess funcion
my_test_img = "DATASETS/archive/PetImages/test/391.jpg"
my_test_img = load(my_test_img)
descriptor = Array{Float64}(undef, 1764, 1)
descriptor[:, 1] = create_descriptor(my_test_img, HOG())


# predict
predicted_label, _ = svmpredict(model, descriptor)
print("This Photo belongs to the class: $(predicted_label[1])")
