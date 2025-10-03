import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Devices:", tf.config.list_physical_devices())

# デバイスのログを表示させる
tf.debugging.set_log_device_placement(True)

a = tf.constant([[1.0, 2.0, 3.0]])
b = tf.constant([[4.0, 5.0, 6.0]])
c = tf.matmul(a, b, transpose_b=True)
print(c)