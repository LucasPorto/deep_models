from tensorflow.python.keras.models import Model
from layers import AffineTransformation
from networks import localisation_network

def spatial_transformer(input_shape, conv_filters, pool_size=2):
    loc_net = localisation_network(input_shape, conv_filters, pool_size)
    affine_tr = AffineTransformation()([loc_net.output, loc_net.inputs[0]])
    model = Model(inputs=loc_net.inputs, outputs=affine_tr)
    return model