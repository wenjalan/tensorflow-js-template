// TensorFlow.js Model Training Template
// For use in competitons, projects, and fooling around.
// Author: Alan Wen (wenjalan@uw.edu)
// 2021-10-27
const tf = require('@tensorflow/tfjs-node');
const data = require('./data.js');

// training options
const DEFAULT_MODEL_SAVE_DIRECTORY = 'file://./model/';
const TRAIN_EPOCHS = 10;
const TRAIN_BATCH_SIZE = 1;

// trains a sequential model and saves it to the given directory
async function train(modelDirectory = DEFAULT_MODEL_SAVE_DIRECTORY) {
    // TODO: load trainXs, trainYs, testXs, testYs from data source
    // default implementation: loads from ./data.js
    const [trainXs, trainYs, testXs, testYs] = await data.load();
    console.log('Loaded training with x shape [' + trainXs.shape + '] and y shape [' + trainYs.shape + '].');

    // initialize the model
    const inputShape = trainXs.shape.slice(1);
    const model = createModel(inputShape);
    console.log('Created model.')
    model.summary();

    // train the model
    console.log('Training model...');
    await model.fit(trainXs, trainYs, {
        epochs: TRAIN_EPOCHS,
        batchSize: TRAIN_BATCH_SIZE,
    });
    console.log('Training completed!');

    // save the model
    await model.save(modelDirectory);
    console.log('Saved model to ' + modelDirectory + '.');
}

// returns a sequential model
function createModel(inputShape) {
    // sequential model
    const model = tf.sequential();

    // TODO: customize model architecture
    model.add(tf.layers.dense({
        inputShape: inputShape,
        units: 10,
        activation: 'relu'
    }));

    model.add(tf.layers.dense({
        units: 1,
        activation: 'softmax'
    }));

    model.compile({
        optimizer: tf.train.adam(),
        loss: 'meanSquaredError',
        metrics: ['accuracy']
    });

    return model;
}

// start
train();