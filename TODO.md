#ToDo

1. Process word vectors

2. Build encode and decode LSTMs

3. Write  model which connects an encoder to a decoder so that an input output pair can be placed in the network and softmax outputs calculated at each *decoding* time step.

4. Once forward pass works, write backward pass to calculate gradients and apply updates to the parameters.

5. Write inference code which stochastically produces an output by sampling according to softmax probabilities at each time step and feeding corresponding word2vec values for the sampled 
6. Write a softening function parameterized by a *temperature* that makes soft max probabilities colder (select less stochastically) or warmer (more entropic).

7. Write beamsearch function to sample approximately the sequence with highest joint probability (controlling for length >= some constant).

8. **The real contribution** Connect end to end so that we can take k exchanges and maximize the probability of all decoded answers given the entire conversation. 

8a. Given a sequence E1 D1 E2 D2, The state of D1 after decoding should be such that it improves the usefulness of E2 such that D2 can give a probable decoding. 


