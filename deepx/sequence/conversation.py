import numpy as np
import theano
import theano.tensor as T
from theano.typed_list import TypedListType as tl

from theanify import theanify, Theanifiable
from deepx.nn import LSTM, Softmax

class ConversationModel(Theanifiable):

    def __init__(self, name, vocab, n_hidden, num_layers=2):
        super(ConversationModel, self).__init__()
        self.name = name
        self.vocab = vocab
        self.word_size = vocab.word_size
        self.vocab_size = vocab.vocab_size
        self.word_matrix = T.as_tensor_variable(vocab.word_matrix)

        self.encoder = LSTM('encoder', self.word_size, n_hidden, num_layers=num_layers)
        self.decoder = LSTM('decoder', self.word_size, n_hidden, num_layers=num_layers)
        self.softmax = Softmax('softmax', n_hidden, self.vocab_size)

        self.parameters = {}

    #@theanify(T.tensor3('X'), T.tensor3('hidden'), T.tensor3('state'))
    def encode(self, X, hidden, state):
        S, N, D = X.shape

        def step(input, previous_hidden, previous_state):
            next_output, next_state = self.encoder.step(input, previous_hidden, previous_state)
            return next_output, next_state

        (encoder_output, encoder_state), _ = theano.scan(step,
                              sequences=[X],
                              outputs_info=[hidden, state],
                              n_steps=S)
        return encoder_output[-1], encoder_state[-1]

    #@theanify(T.iscalar('length'), T.tensor3('output'), T.tensor3('state'), T.vector('start_token', dtype='int64'))
    def decode(self, length, output, state, start_token):

        output = T.unbroadcast(output, 1)
        state = T.unbroadcast(state, 1)

        def step(previous_hidden, previous_state, previous_output):
            word = self.word_matrix[previous_output]
            decoder_output, decoder_state = self.decoder.step(word, previous_hidden, previous_state)
            output = self.softmax.forward(decoder_output[:, -1, :]).argmax(axis=1)
            return decoder_output, decoder_state, output

        (decoder_output, decoder_state, softmax_output), _ = theano.scan(step,
                              sequences=[],
                              outputs_info=[output, state, start_token],
                              n_steps=length)
        return (decoder_output[-1], decoder_state[-1], softmax_output)


    @theanify(T.dtensor4('in_lines'),
              T.dtensor4('out_lines'),
              T.ivector('in_lengths'),
              T.ivector('out_lengths'),
              T.tensor3('hidden'),
              T.tensor3('state'),
              T.vector('start_token', dtype='int64'),
              T.iscalar('batch_size'))
    def seq(self, in_lines, out_lines, in_lengths, out_lengths, hidden, state, start_token, batch_size):
        C = in_lines.shape[0]

        max_length = T.max(out_lengths)

        def step(in_line, in_length, out_length, previous_hidden, previous_state, softmax_output):
            in_line = in_line[:in_length]
            encoder_output, encoder_state = self.encode(in_line,
                                                        previous_hidden,
                                                        previous_state)
            decoder_output, decoder_state, softmax_output = self.decode(
                out_length, encoder_output, encoder_state, start_token)
            return decoder_output, decoder_state, softmax_output

        (decoder_output, decoder_state, softmax_output), _ = theano.scan(step,
                                              sequences=[in_lines, in_lengths, out_lengths],
                                              outputs_info=[hidden, state,
                                                            T.alloc(np.array(0).astype(np.int64), batch_size)],
                                                            n_steps=C)
        return softmax_output
