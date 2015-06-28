import numpy as np
import unittest
from deepx.lstm import LSTM

def logistic(x):
    return 1.0/(1.0 + np.exp(-x))


class TestSimpleLSTM(unittest.TestCase):

    def setUp(self):
        self.lstm = LSTM('lstm', 1, 1, num_layers=1, use_forget_gate=False).compile()
        self.lstm_forget = LSTM('forget', 1, 1, num_layers=1, use_forget_gate=True).compile()
        self.lstm_random = LSTM('random', 10, 10, num_layers=10, use_forget_gate=True,
                                use_input_peep=True,
                                use_forget_peep=True,
                                use_output_peep=True).compile()

    def zero_lstm(self, lstm):
        for param in lstm.parameters:
            lstm.set_parameter_value(param, lstm.get_parameter_value(param) * 0)

    def set_weights(self, lstm, val):
        self.zero_lstm(lstm)
        for param in lstm.parameters:
            if param[0] != "b":
                lstm.set_parameter_value(param, lstm.get_parameter_value(param) + val)


    def test_input_gate(self):
        self.set_weights(self.lstm, 1)
        X = np.ones((1, 1))
        S = np.zeros((1, 1, 1))
        H = np.zeros((1, 1, 1))

        input_gate = logistic(X)
        state = np.tanh(1) * input_gate
        output_gate = logistic(1)
        out = output_gate * np.tanh(state)
        lstm_out, lstm_state = self.lstm.step(X, S, H)

        self.assertEqual(lstm_out, out)

        Wi = self.lstm.get_parameter_value('W_ix')
        self.lstm.set_parameter_value('W_ix', Wi * 0)

        input_gate = 0.5
        state = np.tanh(1) * input_gate
        output_gate = logistic(1)
        out = output_gate * np.tanh(state)
        lstm_out, lstm_state = self.lstm.step(X, S, H)
        self.assertEqual(lstm_out, out)


    def test_output_gate(self):
        self.set_weights(self.lstm, 1)
        X = np.ones((1, 1))
        S = np.zeros((1, 1, 1))
        H = np.zeros((1, 1, 1))

        input_gate = logistic(X)
        state = np.tanh(1) * input_gate
        output_gate = logistic(1)
        out = output_gate * np.tanh(state)
        lstm_out, lstm_state = self.lstm.step(X, S, H)

        self.assertEqual(lstm_out, out)
        self.assertEqual(lstm_state, state)

        Wo = self.lstm.get_parameter_value('W_ox')
        self.lstm.set_parameter_value('W_ox', Wo * 0)

        input_gate = logistic(X)
        state = np.tanh(1) * input_gate
        output_gate = logistic(0)
        out = output_gate * np.tanh(state)
        lstm_out, lstm_state = self.lstm.step(X, S, H)
        self.assertEqual(lstm_out, out)
        self.assertEqual(lstm_state, state)

    def test_forget_gate(self):
        self.set_weights(self.lstm_forget, 1)
        X = np.ones((1, 1))
        S = np.ones((1, 1, 1))
        H = np.zeros((1, 1, 1))

        input_gate = logistic(X)
        forget_gate = logistic(X)
        state = np.tanh(1) * input_gate + forget_gate
        output_gate = logistic(1)
        out = output_gate * np.tanh(state)
        lstm_out, lstm_state = self.lstm_forget.step(X, H, S)

        self.assertEqual(lstm_out, out)

        Wf = self.lstm_forget.get_parameter_value('W_fx')
        self.lstm_forget.set_parameter_value('W_fx', Wf * 0)

        input_gate = logistic(X)
        forget_gate = 0.5
        state = np.tanh(1) * input_gate + forget_gate
        output_gate = logistic(1)
        out = output_gate * np.tanh(state)
        lstm_out, lstm_state = self.lstm_forget.step(X, H, S)

        self.assertEqual(lstm_out, out)

    def test_save_load(self):
        lstm = LSTM.load(self.lstm_random.state())
        for param in lstm.parameters:
            print param
            self.assertTrue((lstm.get_parameter_value(param) == self.lstm_random.get_parameter_value(param)).all())




if __name__ == "__main__":
    unittest.main()
