from recommenders.models.newsrec.models.nrms import NRMSModel
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

K = tf.keras.backend


# Modified from https://stackoverflow.com/questions/53867351/how-to-visualize-attention-weights
class CharVal(object):
    def __init__(self, char, val):
        self.char = char
        self.val = val

    def __str__(self):
        return str(round(self.val,4)) # TODO funktioniert noch nicht so wie sie soll sie print(char_df)

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def color_charvals(s):
    r = 255-int(s.val*255)
    color = rgb_to_hex((255, r, r))
    return 'background-color: %s' % color


def visualize_attention(att_weight : np.array, scale = True):
    sum_ = np.sum(att_weight)
    c_vals = []

    # Scale for better visualization
    scaling_factor = 1

    print(np.max(att_weight[:,:,0]))

    if scale:
        while(np.max(att_weight[:,:,0])*scaling_factor < 1):
            scaling_factor *= 10
        print(f"Results are scaled by factor {scaling_factor}")  
    
    for b in range(att_weight.shape[0]):
        dict_ = dict(zip(np.arange(0,att_weight.shape[1],1,int),att_weight[b,:,0]))

        # before the prediction i supposed you tokenized text
        # you need to match each char and attention

        
        # TODO GGF die Summe entfernen
        char_vals = [CharVal(c, (v/sum_)*scaling_factor) for c, v in dict_.items()]
        c_vals.append(char_vals)
    
    
    char_df = pd.DataFrame(c_vals,columns=np.arange(1,(att_weight.shape[1] + 1),1,int))
    char_df.style.set_properties(**{'text-align': 'center'})

    # apply coloring values
    char_df = char_df.style.applymap(color_charvals)
    display(char_df)


def get_attention_nrms(model : NRMSModel,user_input):
    """Returns the attention weights for each news in the history (inside the user encoder)

    Args:
        model (NRMSModel): trained NRMS model
        user_input (_type_): User input with a form (batch_size x max_hist_size x max_title_length)

    Returns:
        _type_: Array of weights with size (batch_size x max_hist_size x 1)
    """



    # Input runs to the two first layers
    inputs = model.userencoder.layers[0](user_input)
    inputs = model.userencoder.layers[1](inputs)

    # Weights for attention mechanism
    w_ = model.userencoder.layers[1].weights[4]
    b_ = model.userencoder.layers[1].weights[5]
    q_ = model.userencoder.layers[1].weights[6]

    # tanh(w.x + b) . q
    attention = K.tanh(K.dot(inputs, w_) + b_)
    attention = K.dot(attention, q_)

    attention = K.squeeze(attention, axis=2)

    # Mask out every padding news titel e.g. User has history size 12 and max history size is 100 than the first 88 values are masked out
    assert len(user_input.shape) == 3 , print("User input has to be of shape (batchsize, max_hist_size, max_title_size)")
    mask = np.count_nonzero(user_input != 0,axis=2)
    mask = mask != 0
    if mask is None:
            attention = K.exp(attention)
    else:
            attention = K.exp(attention) * K.cast(mask, dtype="float32")
    
    # Attention weights are basically probabilies 
    attention = K.exp(attention)

    attention_weight = attention / (
        K.sum(attention, axis=-1, keepdims=True) + K.epsilon()
    )
    
    attention_weight = K.expand_dims(attention_weight)

    return attention,attention_weight