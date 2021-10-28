// TensorFlow.js Model Loading Template
// For use in competitons, projects, and fooling around.
// Author: Alan Wen (wenjalan@uw.edu)
// 2021-10-27
const tf = require('@tensorflow/tfjs-node');

const DEFAULT_MODEL_LOAD_DIRECTORY = 'file://./model/model.json';

// loads the model from the specified directory
// if no directory is specified, the default directory is used
async function loadModel(modelDirectory = DEFAULT_MODEL_LOAD_DIRECTORY) {
    console.log('Loading model from ' + modelDirectory + '...');
    const model = await tf.loadLayersModel(modelDirectory);
    model.summary();
    return model;
}

exports.loadModel = loadModel;