import tensorflow as tf


class VGG16(tf.keras.Model):

    def __init__(self, keep_prob, num_classes):
        super().__init__()
        self.keep_prob = keep_prob
        self.num_classes = num_classes

        def _block(filters, n_convs, name):
            layers = []
            for i in range(n_convs):
                layers.append(tf.keras.layers.Conv2D(
                    filters, 3, padding="same", activation="relu",
                    name="{}_conv{}".format(name, i + 1)))
            layers.append(tf.keras.layers.MaxPool2D(2, 2, name="{}_pool".format(name)))
            return layers

        self._conv_layers = (
            _block(64, 2, "block1")
            + _block(128, 2, "block2")
            + _block(256, 3, "block3")
            + _block(512, 3, "block4")
            + _block(512, 3, "block5")
        )
        self._flatten = tf.keras.layers.Flatten()
        self._fc = tf.keras.layers.Dense(num_classes, name="fc")

    def _build_model(self, inputs):
        x = inputs
        for layer in self._conv_layers:
            x = layer(x)
        x = self._flatten(x)
        return self._fc(x)

    def call(self, inputs, training=False):
        return self._build_model(inputs)
