const tf = require("@tensorflow/tfjs-node-gpu");

// Basics
console.log("\nBasics");
tensor = tf.scalar(1);
tensor.print();

tensor = tf.tensor([1, 2, 3, 4]);
tensor.print();

tensor = tf.tensor([1, 2, 3, 4], [2, 2]);
tensor.print();

tensor = tf.tensor([1, 2, 3, 4], [2, 2], "int32");
tensor.print();

tensor = tf.tensor([1, 2, 3, 4]);

// Get Data
console.log("\nGet Data");
// tensor.data().then(result => console.log(result));
console.log(tensor.dataSync());

// Get Elements from Tensor
console.log("\nGet Elements from Tensor");
console.log(tensor.get(0));
console.log(tensor.get(1));
console.log(tensor.get(2));
console.log(tensor.get(3));

// Creating an equation
console.log("\nCreating an equation");
a = tf.tensor(0.1).variable();
b = tf.tensor(0.1).variable();
c = tf.tensor(0.1).variable();

const equation = x =>
  a
    .mul(x.mul(x))
    .add(b.mul(x))
    .add(c);

equation(tf.scalar(1)).print();

// Matrices
console.log("\nMatrices");
matrixA = tf.tensor([1, 2, 3, 4], [2, 2]);
matrixB = tf.tensor([5, 6, 7, 8], [2, 2]);

// Matrix Addition
console.log("\nMatrix Addition");
matrixA.add(matrixB).print();

// Matrix Multiplication
console.log("\nMatrix Multiplication");
matrixA.matMul(matrixB).print();

// Transpose
console.log("\nTranspose");
matrixA.transpose().print();

// Memory Status and Manual Cleanup
console.log("\nMemory Status and Manual Cleanup");
temp = tf.scalar(0);
console.log("--BEFORE--", tf.memory());
temp.dispose();
console.log("--AFTER--", tf.memory());

// Automatic Cleanup
console.log("\nMemory Status and Automatic Cleanup");
tf.tidy(() => {
  temp = tf.scalar(0);
  console.log("--BEFORE--", tf.memory());
});
console.log("--AFTER--", tf.memory());
