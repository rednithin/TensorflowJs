const tf = require("@tensorflow/tfjs-node");

const { dense } = tf.layers;
const { meanSquaredError } = tf.losses;
const { sgd } = tf.train;

xs = tf.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]);
ys = tf.tensor([[0], [1], [1], [0]]);

model = tf.sequential({
  layers: [
    dense({ inputShape: [2], units: 16, activation: "relu" }),
    dense({ units: 1, activation: "sigmoid" })
  ]
});

model.compile({
  loss: meanSquaredError,
  optimizer: sgd(0.5), // To customize extract from tf.train
  metrics: ["binaryAccuracy"]
});

model.fit(xs, ys, { batchSize: 1, epochs: 100 }).then(history => {
  model.predict(xs).print();
});
