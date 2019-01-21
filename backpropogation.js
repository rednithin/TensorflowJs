const fs = require("fs");
const tf = require("@tensorflow/tfjs-node-gpu");

const { dense } = tf.layers;
const {
  sigmoidCrossEntropy,
  softmaxCrossEntropy,
  meanSquaredError
} = tf.losses;
const { sgd, adam } = tf.train;

let dataset = fs
  .readFileSync("Backpropogation.csv")
  .toString()
  .split("\n")
  .map(row => row.split(","));

dataset.pop();
dataset = tf.tensor(dataset, [199, 8], "float32");

const [inputs, targets] = tf.split(dataset, [7, 1], 1);
const oneHots = tf.oneHot(targets.toInt(), 3).reshape([199, 3]);

inputs.print();
oneHots.print();

const model = tf.sequential({
  layers: [
    dense({ inputShape: [7], units: 16, activation: "relu" }),
    dense({ units: 8, activation: "relu" }),
    dense({ units: 3, activation: "softmax" })
  ]
});

model.compile({
  loss: meanSquaredError,
  optimizer: adam(0.01),
  metrics: ["accuracy", "categoricalCrossentropy", "categoricalAccuracy"]
});

model
  .fit(inputs, oneHots, {
    shuffle: true,
    epochs: 100
  })
  .then(async () => {
    await model.save("file://Model");
  });
