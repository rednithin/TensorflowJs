const tf = require("@tensorflow/tfjs-node-gpu");

m = tf.scalar(Math.random()).variable();
b = tf.scalar(Math.random()).variable();

const f = x => m.mul(x).add(b);

const lossFunction = (pred, actual) => {
  return pred
    .sub(actual)
    .square()
    .mean();
};

const learningRate = 0.02;
const optimizer = tf.train.sgd(learningRate);

x = tf.tensor([1, 2, 3, 4, 5]);
y = tf.tensor([3, 6, 9, 12, 15]);

for (let i = 0; i < 501; i++) {
  // optimizer.minimize(() => lossFunction(f(x), y)); - Modifies all variables in the system.
  tf.tidy(() => {
    optimizer.minimize(() => lossFunction(f(x), y), true, [m, b]); // Modifies only m and b
    if (i % 100 == 0) {
      console.log("\nSlope");
      m.print();
      console.log("Y Intercept");
      b.print();
      console.log("Predictions");
      f(x).print();
    }
  });
}

console.log(tf.memory().numTensors);
