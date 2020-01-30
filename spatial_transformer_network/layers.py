import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Layer


class AffineTransformation(Layer):
    def __init__(self):
        super(AffineTransformation, self).__init__()

        # four neighboring pixels
        shift_vectors = tf.convert_to_tensor([[0, 1, 1],
                                              [1, 0, 1],
                                              [0, 0, 0]])

        shift_vectors = tf.cast(shift_vectors, tf.float32)
        self.shift_vectors = tf.reshape(shift_vectors, (3, 3))

    def build(self, input_shape):
        theta_shape, img_shape = input_shape
        _, rows, cols, _ = img_shape
        x_coords = tf.linspace(-1.0, 1.0, cols)
        y_coords = tf.linspace(-1.0, 1.0, rows)
        x_grid, y_grid = tf.meshgrid(x_coords, y_coords)
        target_grid = tf.stack([x_grid, y_grid, tf.ones_like(x_grid)])
        self.target_vectors = tf.reshape(target_grid, [3, -1])

        rows_f = tf.cast(rows, tf.float32)
        cols_f = tf.cast(cols, tf.float32)
        self.rescaling_matrix = tf.convert_to_tensor([[(cols_f - 1) / 2.0, 0, (cols_f - 1) / 2.0],
                                                      [0, (rows_f - 1) / 2.0, (rows_f - 1) / 2.0],
                                                      [0, 0, 1]], tf.float32)

        self.clip_zero = tf.convert_to_tensor([0, 0, 0], tf.float32)
        self.clip_max = tf.convert_to_tensor([cols_f - 1, rows_f - 1, 1], tf.float32)

    def call(self, inputs, **kwargs):
        theta, image = inputs
        n = tf.shape(theta)[0]
        theta = tf.concat([theta, tf.cast(tf.tile([[0, 0, 1]], [n, 1]), tf.float32)], 1)
        affine_transformation = tf.reshape(theta, [-1, 3, 3])
        source_vectors = tf.matmul(affine_transformation, self.target_vectors)
        interpolated = self._sampling_kernel(source_vectors, image)
        return interpolated

    def _sampling_kernel(self, source_vectors, image):
        # Rescaling from [-1, 1] to [0, row] X [0, col]
        source_rescaled = tf.matmul(tf.expand_dims(self.rescaling_matrix, axis=0), source_vectors)
        source_index = tf.floor(source_rescaled)

        neighbors = []
        for i in range(3):
            shift = tf.expand_dims(self.shift_vectors[:, i:i+1], axis=0)
            shifted = tf.add(source_index, shift)
            neighbors.append(shifted)

        neighbors.append(source_index)
        neighbors = tf.stack(neighbors, axis=-1)
        neighbors = tf.clip_by_value(neighbors,
                                     tf.reshape(self.clip_zero, (1, 3, 1, 1)),
                                     tf.reshape(self.clip_max, (1, 3, 1, 1)))

        # (x, y, 1) -> (x, y)
        source_rescaled = source_rescaled[:, 0:2, :]
        neighbors = neighbors[:, 0:2, :, :]
        indices = tf.cast(neighbors, tf.int32)

        # Storing interpolation coefficients max(0, 1 - |x_s - m|)*max(0, 1 - |y_s - n|)
        deltas = tf.maximum(0.0, 1 - tf.abs(tf.expand_dims(source_rescaled, axis=-1) - neighbors))
        interp_coefs = tf.reduce_prod(deltas, axis=1)

        batch_size = tf.shape(image)[0]
        batch_idx = tf.range(batch_size)
        ind_shape = tf.shape(indices)[1:]
        batch_idx = tf.tile(tf.reshape(batch_idx, (batch_size, 1, 1)),
                            (1, ind_shape[1], ind_shape[2]))

        col_indices, row_indices = tf.unstack(indices, axis=1)
        indices = tf.stack([batch_idx, row_indices, col_indices], axis=1)

        # Rearranging (batch, [batch_id, row, col], grid_index, neighbor i)
        # to (batch, grid_index, [b, row, col]) for gather_nd
        pixel_neighbors = []
        for i in range(4):
            indices_neighbor_i = indices[:, :, :, i]
            indices_neighbor_i = tf.transpose(indices_neighbor_i, [0, 2, 1])
            pixel_neighbor_i = tf.gather_nd(image, indices_neighbor_i)
            pixel_neighbors.append(pixel_neighbor_i)

        # Multiplying input image values with interpolation coefficients (Equation 5)
        pixel_neighbors = tf.concat(pixel_neighbors, axis=-1)
        interpolated = tf.multiply(tf.cast(pixel_neighbors, tf.float32), interp_coefs)
        interpolated = tf.reduce_sum(interpolated, axis=-1)
        interpolated = tf.reshape(interpolated, tf.shape(image))
        interpolated = tf.clip_by_value(interpolated, 0.0, 1.0)

        return interpolated









































