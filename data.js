// TensorFlow.js Data Load Template
// For use in competitons, projects, and fooling around.
// Author: Alan Wen (wenjalan@uw.edu)
// 2021-10-27
const tf = require('@tensorflow/tfjs-node');

// loads data from a source
// returns Tensors to train a model on
async function load() {
    // TODO: Implement your own data loading logic here
    const trainXs = tf.zeros([10, 1]);
    const trainYs = tf.zeros([10, 1]);
    const testXs = tf.zeros([10, 1]);
    const testYs = tf.zeros([10, 1]);
    return [trainXs, trainYs, testXs, testYs];
}

exports.load = load;