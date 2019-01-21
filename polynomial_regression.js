const tf = require("@tensorflow/tfjs-node-gpu");

a = tf.scalar(Math.random()).variable();
b = tf.scalar(Math.random()).variable();
c = tf.scalar(Math.random()).variable();

const f = x =>
  a
    .mul(x.square())
    .add(b.mul(x))
    .add(c);

const lossFunction = (pred, actual) => {
  return pred
    .sub(actual)
    .square()
    .mean();
};

const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

x = tf.tensor([0.1, 0.4, 0.6, 0.8, 1]);
y = tf.tensor([0, 0.3, 1, 0.3, 0]);

for (let i = 0; i < 10000; i++) {
  tf.tidy(() => {
    optimizer.minimize(() => lossFunction(f(x), y));
    console.log("Predictions");
    f(x).print();
  });
}

console.log(tf.memory().numTensors);
