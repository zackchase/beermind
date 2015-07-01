import numpy as np
import theano
import theano.tensor as T

from theanify import theanify
from deepx.nn import LSTM, LSTM2, Softmax, ParameterModel

class ConversationModel(ParameterModel):

    def __init__(self, name, vocab, n_hidden, num_layers=2):
        super(ConversationModel, self).__init__(name)
        self.name = name
        self.vocab = vocab
        self.word_size = vocab.word_size
        self.vocab_size = vocab.vocab_size
        self.word_matrix = T.as_tensor_variable(vocab.word_matrix)

        self.encoder = LSTM('encoder', self.word_size, n_hidden, num_layers=num_layers)
        self.decoder = LSTM2('decoder', self.word_size, n_hidden, num_layers=num_layers)
        self.softmax = Softmax('softmax', n_hidden, self.vocab_size)

        self.average_gradient = [theano.shared(p.get_value() * 0) for p in self.get_parameters()]
        self.average_rms = [theano.shared(p.get_value() * 0) for p in self.get_parameters()]
        self.parameter_update = [theano.shared(p.get_value() * 0) for p in self.get_parameters()]

    #@theanify(T.tensor3('X'), T.tensor3('hidden'), T.tensor3('state'))
    def encode(self, X, hidden, state):
        S, N, D = X.shape

        hidden = T.unbroadcast(hidden, 1)
        state = T.unbroadcast(state, 1)

        def step(input, previous_hidden, previous_state):
            print "Encoding step", self.encoder.name
            next_output, next_state = self.encoder.step(input, previous_hidden, previous_state)
            print self.encoder.name
            print "Done with encoding step"
            return next_output, next_state

        (encoder_output, encoder_state), _ = theano.scan(step,
                              sequences=[X],
                              outputs_info=[hidden, state],
                              n_steps=S)
        return encoder_output[-1], encoder_state[-1]

    #@theanify(T.iscalar('length'), T.tensor3('output'), T.tensor3('state'), T.vector('start_token', dtype='int64'))
    def decode(self, length, hidden, state, start_token):

        hidden = T.unbroadcast(hidden, 1)
        state = T.unbroadcast(state, 1)

        def step(previous_hidden, previous_state, previous_output):
            word = self.word_matrix[previous_output.argmax(axis=1)]
            decoder_output, decoder_state = self.decoder.step(word, previous_hidden, previous_state)
            output = self.softmax.forward(decoder_output[:, -1, :])
            return decoder_output, decoder_state, output

        (decoder_output, decoder_state, softmax_output), _ = theano.scan(step,
                              sequences=[],
                              outputs_info=[hidden, state, start_token],
                              n_steps=length)
        return (decoder_output[-1], decoder_state[-1], softmax_output)


    #@theanify(T.dtensor3('in_line'),
              #T.iscalar('out_length'),
              #T.matrix('start_token', dtype='int64'),
              #)
    def seq2seq(self, in_line, out_length, start_token):
        _, N, D = in_line.shape
        L = self.encoder.num_layers
        H = self.encoder.n_hidden
        hidden = T.alloc(np.array(0).astype(theano.config.floatX), N, L, H)
        state = T.alloc(np.array(0).astype(theano.config.floatX), N, L, H)

        print "Encoding"
        encoder_hidden, encoder_state = self.encode(in_line,
                                                    hidden,
                                                    state)
        print "Decoding"
        decoder_hidden, decoder_state, softmax_output = self.decode(
            out_length, encoder_hidden, encoder_state, start_token)

        return softmax_output

    #@theanify(T.tensor3('in_line'),
              #T.tensor3('out_line'),
              #T.matrix('start_token'),
              #)
    def loss(self, in_line, out_line, start_token):
        out_length = out_line.shape[0]
        out_pred = self.seq2seq(in_line, out_length, start_token)
        return T.sum(T.nnet.categorical_crossentropy(out_pred, out_line))

    #@theanify(T.tensor3('in_line'),
              #T.tensor3('out_line'),
              #T.matrix('start_token'),
              #)
    def gradient(self, in_line, out_line, start_token, clip=5):
        loss = self.loss(in_line, out_line, start_token)
        return T.grad(cost=loss, wrt=self.get_parameters())

    @theanify(T.tensor3('in_line'),
              T.tensor3('out_line'),
              T.matrix('start_token'),
              updates='rmsprop_updates'
              )
    def rmsprop(self, in_line, out_line, start_token):
        return self.loss(in_line, out_line, start_token)

    def rmsprop_updates(self, in_line, out_line, start_token):
        grads = self.gradient(in_line, out_line, start_token)
        next_average_gradient = [0.95 * avg + 0.05 * g for g, avg in zip(grads, self.average_gradient)]
        next_rms = [0.95 * rms + 0.05 * (g ** 2) for g, rms in zip(grads, self.average_rms)]
        next_parameter = [0.9 * param_update - 1e-4 * g / T.sqrt(rms - avg ** 2 + 1e-4)
                          for g, avg, rms, param_update in zip(grads,
                                                               self.average_gradient,
                                                               self.average_rms,
                                                               self.parameter_update)]

        average_gradient_update = [(avg, next_avg) for avg, next_avg in zip(self.average_gradient,
                                                                            next_average_gradient)]
        rms_update = [(rms, rms2) for rms, rms2 in zip(self.average_rms,
                                                               next_rms)]
        next_parameter_update = [(param, param_update) for param, param_update in zip(self.parameter_update,
                                                                                      next_parameter)]

        updates = [(p, p + param_update) for p, param_update in zip(self.parameters(), next_parameter)]

        return updates + average_gradient_update + rms_update + next_parameter_update

    def get_parameters(self):
        return self.encoder.get_parameters() + self.decoder.get_parameters() + self.softmax.get_parameters()
