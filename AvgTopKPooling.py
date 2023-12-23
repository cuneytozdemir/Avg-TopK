class AvgTopKPooling(tf.keras.layers.Layer):
    def __init__(self, ksize=3,kk=5):
        super(AvgTopKPooling, self).__init__()
        self.ksize = ksize
        self.kk = kk
        
    def call(self, inputs):                  
        k_size=self.ksize
        channel = inputs.shape[3]
        x_patches = tf.image.extract_patches(inputs,
                        sizes=[1,k_size,k_size,1],
                        strides=[1,k_size,k_size,1],
                        rates=[1,1,1,1],
                        padding='VALID')    
        
        return tf.concat([tf.reduce_mean(tf.math.top_k(x_patches[:,:,:,c::channel],k=self.kk).values,keepdims=True, axis=-1) for c in range(channel)], axis=-1)
