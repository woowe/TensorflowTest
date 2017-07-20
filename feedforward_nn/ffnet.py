import tensorflow as tf

class FFNet:
    def __init__(self, input_size, output_size, hidden_desc, regression = False):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_desc = hidden_desc
        self.regression = regression
        self.weights = []
        self.activations = []
        self.x = tf.placeholder(tf.float32, [None, input_size], name="ffnet_x")
        self.y = tf.placeholder(tf.float32, name="ffnet_y")
        self._init_weights()
        self._build_graph()
    
    def _init_weights(self):
        self.weights.append({ 'weights': tf.Variable(tf.random_normal([self.input_size, self.hidden_desc[0]['hidden_nodes']])),
                              'biases': tf.Variable(tf.random_normal([self.hidden_desc[0]['hidden_nodes']]))})
        
        if len(self.hidden_desc[1:]) >= 1:
            cnt = 0
            for desc in self.hidden_desc[1:]:
                i_size = self.hidden_desc[cnt]['hidden_nodes']
                self.weights.append({ 'weights': tf.Variable(tf.random_normal([i_size, desc['hidden_nodes']])),
                                      'biases': tf.Variable(tf.random_normal([desc['hidden_nodes']]))})
                cnt += 1
        
        self.weights.append({ 'weights': tf.Variable(tf.random_normal([self.hidden_desc[len(self.hidden_desc) - 1]['hidden_nodes'], self.output_size])),
                              'biases': tf.Variable(tf.random_normal([self.output_size]))})
    
    def _build_graph(self):
        activ_func = self.hidden_desc[0]['activation']
        self.activations.append(
            activ_func(tf.matmul(self.x, self.weights[0]['weights']) + self.weights[0]['biases'])
        )
        
        if len(self.weights[1:-1]) >= 1:
            cnt = 1
            for weight in self.weights[1:-1]:
                activ_func = self.hidden_desc[cnt]['activation']
                prev_activ = self.activations[len(self.activations) - 1]
                self.activations.append(
                    activ_func(tf.matmul(prev_activ, weight['weights']) + weight['biases'])
                )
                cnt += 1
        
        weights_len = len(self.weights)
        prev_activ = self.activations[len(self.activations) - 1]
        self.logits = tf.matmul(prev_activ, self.weights[weights_len - 1]['weights']) + self.weights[weights_len - 1]['biases']
    
    def predict(self, x):
        prediction = tf.nn.softmax(self.logits)
        pred_argmax = tf.argmax(prediction, 1)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            pred, argmax = sess.run([prediction, pred_argmax], feed_dict = {self.x: x})
            return pred, argmax
    
    def train(self, training_data, batch_size, epochs, debug = False):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                epoch_cost = 0
                for _ in range(int(training_data.train.num_examples/batch_size)):
                    x, y = training_data.train.next_batch(batch_size)
                    _, c = sess.run([optimizer, cost], feed_dict = {self.x: x, self.y: y})
                    epoch_cost += c
                if debug:
                    print('Epoch: ', (epoch + 1), '/', epochs, ' Epoch cost: ', epoch_cost)
            if debug:
                correct = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                print('Accuracy', accuracy.eval({self.x:training_data.test.images,
                                                 self.y:training_data.test.labels}))