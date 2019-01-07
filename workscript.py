input01 = tf.keras.Input(shape=(number_of_frames,720 ,720 , 3,))
avg01 = layers.AvgPool3D(pool_size=(1,5,5),padding="same")(input01)
dropout01 = layers.Dropout(rate=0.2)(avg01)
avg02 = layers.AvgPool3D(pool_size=(1,2,2),padding="same")(dropout01)
model01 = tf.keras.Model(inputs=input01, outputs=avg02)

#model01 = tf.keras.Sequential()
#model01.add(layers.AvgPool3D(pool_size=(1,5,5),padding="same"))
#model01.add(layers.Dropout(rate=0.2))
#model01.add(layers.AvgPool3D(pool_size=(1,2,2),padding="same"))

#model01.add(layers.Flatten())
#model01.add(layers.Dense(5184,activation="relu"))
#model01.compile(loss='mean_squared_error', optimizer='sgd')


#model01.add(layers.Conv2D(3, (3, 3),padding='same'))
#model01.add(layers.Dense(3,activation="relu"))

#model01.add(layers.Dropout(rate=0))

def reshape_Pix(x):
    #return tf.keras.backend.reshape(x,shape=[1,1,720,720,3])
    return tf.keras.backend.reshape(x,shape=[1,720,720,3,])
input01 = tf.keras.Input(shape=(number_of_frames,720 ,720 , 3,))
base01= layers.Lambda(getBase)(input01)
avg01 = layers.AvgPool3D(pool_size=(1,5,5),padding="same")(input01)
avg02 = layers.AvgPool3D(pool_size=(1,2,2),padding="same")(avg01)
reshape01 = tf.keras.layers.Reshape([518400,3,])(avg02)
dense01=layers.Dense(512,activation="relu")(reshape01)
dropout01 = layers.Dropout(rate=0.2)(dense01)
dense02=layers.Dense(16,activation="relu")(dropout01)
dropout02 = layers.Dropout(rate=0.2)(dense01)
dense03=layers.Dense(5,activation="relu")(dropout02)
dense04=tf.keras.layers.Dense(1555200,activation="relu")(dense03)
lambda01 = layers.Lambda(reshape_Pix)(dense04)

input02 = tf.keras.Input(shape=(1,720 ,720 , 3,))
merge01 = tf.keras.layers.Multiply()([input02,lambda01])
#Add a merge layer of multipy
#lambda01 = layers.Lambda(shiftPixels, arguments={'y':base01})(dense03)

model = tf.keras.Model(inputs=[input01,input02], outputs=merge01)
#model.compile(loss='mean_squared_error', optimizer='sgd')
model.compile(loss=nextFrameLoss, optimizer='sgd')



'''
print(tf.VERSION)
print(tf.keras.__version__)

model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(64, activation='relu'))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(layers.Dense(10, activation='softmax'))


# Create a sigmoid layer:
layers.Dense(64, activation='sigmoid')
# Or:
layers.Dense(64, activation=tf.sigmoid)
# A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))
# A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))
# A linear layer with a kernel initialized to a random orthogonal matrix:
layers.Dense(64, kernel_initializer='orthogonal')
# A linear layer with a bias vector initialized to 2.0s:
layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))
'''

'''
# Training
model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(64, activation='relu'),
# Add another:
layers.Dense(64, activation='relu'),

# Add a softmax layer with 10 output units:
layers.Dense(10, activation='softmax')])


model.compile(loss='mean_squared_error', optimizer='sgd')



data = np.random.random((1000, 64))
labels = np.random.random((1000, 10))

print(labels)

model.fit(data, labels, epochs=10, batch_size=32)

tf.keras.models.save_model(
    model,
    "saves/savetest.h5",
    overwrite=True,
    include_optimizer=True
)
'''