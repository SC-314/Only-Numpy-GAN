np.random.seed(42)
GW0 = Glorot_init(100,7*7*128) # Output from 7*7*n will be n features maps of 7 by 7
GW1 = Glorot_init(128, 128, 3) # (output size, input size, filter size)
GW2 = Glorot_init(64, 128, 3)
GW3 = Glorot_init(64, 64, 5)
GW4 = Glorot_init(32, 64, 5)
GW5 = Glorot_init(32, 32, 5)
GW6 = Glorot_init(1, 32, 6)

DW0 = Glorot_init(64,1,3)
DW2 = Glorot_init(64,64,3) # Output from n will be a vector of size n * 7 * 7
DW4 = Glorot_init(64*7*7, 1)

total = 0
correct = 0
graph = []
counter = 0

G, D = [], []
store_fake_images = []
store_constant_images = []

np.random.seed(42)
test_noise = list(np.random.normal(0,1, (6, 100)))

data = []
sols = []
n_classes = 10
for i in range(n_classes):
    for j in range(200): # Change the file address to your own, with each image having image_n.jpg
        data.append((np.array(Image.open(f'C:/Users/Sam/Desktop/Generative Adversarial Network/Data/trainingSet/trainingSet/{i}/image_{j}.jpg').convert('L'), dtype=np.float32))/255-0.10)
