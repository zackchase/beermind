import numpy as np
import theano
import theano.tensor as T

theano.config.on_unused_input = 'ignore'

from theanify import theanify
from deepx.nn import Softmax, ParameterModel

from encdec import Encoder, Decoder

class ConversationModel(ParameterModel):

    def __init__(self, name, vocab, n_hidden, max_seq_length, num_layers=2):
        super(ConversationModel, self).__init__(name)
        self.name = name
        self.vocab = vocab
        self.word_size = vocab.word_size
        self.vocab_size = vocab.vocab_size
        self.word_matrix = T.as_tensor_variable(vocab.word_matrix)
        self.max_seq_length = max_seq_length

        self.encoder = Encoder('encoder', self.word_size, n_hidden, max_seq_length, num_layers=num_layers)
        softmax = Softmax('softmax', n_hidden, self.vocab_size)
        self.decoder = Decoder('decoder', self.word_size, n_hidden, max_seq_length, softmax, self.word_matrix,
                               num_layers=num_layers)

        self.average_gradient = [theano.shared(p.get_value() * 0) for p in self.get_parameters()]
        self.average_rms = [theano.shared(p.get_value() * 0) for p in self.get_parameters()]
        self.parameter_update = [theano.shared(p.get_value() * 0) for p in self.get_parameters()]


    #@theanify(T.tensor4('in_lines'), T.tensor3('in_masks'), T.tensor3('out_masks'), T.matrix('start_token'))
    def seq2seq2seq(self, in_lines, in_masks, out_masks, start_token):
        C, N, S, D = in_lines.shape

        L = self.encoder.num_layers
        H = self.encoder.n_hidden

        hidden = T.alloc(np.array(0).astype(theano.config.floatX), N, L, H)
        state = T.alloc(np.array(0).astype(theano.config.floatX), N, L, H)
        hidden = T.unbroadcast(hidden, 1)
        state = T.unbroadcast(state, 1)
        output = T.alloc(np.array(0).astype(theano.config.floatX), self.max_seq_length, N, self.vocab_size)

        def step(in_line, in_mask, out_mask, previous_hidden, previous_state, previous_output):
            encoder_hidden, encoder_state = self.encoder.encode(in_line, in_mask, previous_hidden, previous_state)
            decoder_output, decoder_hidden, decoder_state = self.decoder.decode(start_token, out_mask, encoder_hidden, encoder_state)
            return decoder_hidden, decoder_state, decoder_output

        rval, _ = theano.scan(step,
                              sequences = [in_lines, in_masks, out_masks],
                              outputs_info = [hidden, state, output],
                              n_steps=C
                              )
        return rval

    def seq2seq(self, in_line, in_mask, out_mask, start_token):
        N, S, D = in_line.shape

        L = self.encoder.num_layers
        H = self.encoder.n_hidden

        previous_hidden = T.alloc(np.array(0).astype(theano.config.floatX), N, L, H)
        previous_state = T.alloc(np.array(0).astype(theano.config.floatX), N, L, H)

        encoder_hidden, encoder_state = self.encoder.encode(in_line, in_mask, previous_hidden, previous_state)

        decoder_output, decoder_hidden, decoder_state = self.decoder.decode(start_token, out_mask, encoder_hidden, encoder_state)
        return decoder_output

    #@theanify(T.tensor3('in_line'), T.matrix('in_mask'), T.matrix('out_line'), T.matrix('out_mask'), T.matrix('start_token'))
    def loss(self, in_lines, in_masks, out_lines, out_masks, start_token):
        C, N, _, _ = out_lines.shape
        out_pred = self.seq2seq2seq(in_lines, in_masks, out_masks, start_token)
        out_pred *= out_masks
        out_pred = out_pred.reshape((C * self.max_seq_length * N, self.vocab_size))
        out_line = out_lines.reshape((C * self.max_seq_length * N, self.vocab_size))
        return T.sum(T.nnet.categorical_crossentropy(out_pred, out_line))

    def gradient(self, in_line, in_mask, out_line, out_mask, start_token):
        loss = self.loss(in_line, in_mask, out_line, out_mask, start_token)
        return T.grad(cost=loss, wrt=self.get_parameters())

    @theanify(T.tensor4('in_lines'), T.tensor3('in_masks'), T.tensor4('out_lines'), T.tensor3('out_masks'), T.matrix('start_token'), updates="rmsprop_updates")
    def rmsprop(self, in_lines, in_masks, out_lines, out_masks, start_token):
        return self.loss(in_lines, in_masks, out_lines, out_masks, start_token)

    def rmsprop_updates(self, in_lines, in_masks, out_lines, out_masks, start_token):
        grads = self.gradient(in_lines, in_masks, out_lines, out_masks, start_token)
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

        updates = [(p, p + param_update) for p, param_update in zip(self.get_parameters(), next_parameter)]

        return updates + average_gradient_update + rms_update + next_parameter_update

    def get_parameters(self):
        return self.encoder.get_parameters() + self.decoder.get_parameters()
