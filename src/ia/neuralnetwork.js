import 'babel-polyfill';
import * as tf from '@tensorflow/tfjs';
import { CANVAS_WIDTH, CANVAS_HEIGHT } from '../component/game/constants';
import { Runner } from '../component/game';

let runner = null;
function setup() {
  runner = new Runner('.game', {
    DINO_COUNT: 1,
    onReset: handleReset,
    onCrash: handleCrash,
    onRunning: handleRunning
  });
  window.runner = runner;
  runner.init();
}

let firstTime = true; 

 function handleReset(dinos) {
  // console.log(dinos); 
  const dino = dinos[0]; 
  if (firstTime) {
    firstTime = false;
    dino.model = tf.sequential();
    // dino.model.init();
    dino.model.add(tf.layers.dense({
      inputShape:[4],
      activation:'sigmoid',
      units:6
    }))
    dino.model.add(tf.layers.dense({
      inputShape:[6],
      activation:'sigmoid',
      units:1
    }))
    dino.model.compile({
      loss:'meanSquaredError',
      optimizer : tf.train.adam(0.1)
    })
    dino.training = {
      inputs: [],
      labels: []
    };
  } else {
    console.info('Aprendendo');
    console.info(tf.tensor2d(dino.training.inputs))
    dino.model.fit(tf.tensor2d(dino.training.inputs), tf.tensor2d(dino.training.labels));
    // console.log(runner.playCount); 
    dino.model.save(`localstorage://my-model-dino${runner.playCount}`);
  }
}

/**
 * documentation
 * @param {object} dino
 * @param {object} state
 * returns a promise resolved with an action
 */

function handleRunning( dino, state ) {
  return new Promise((resolve) => {
    if (!dino.jumping) {
      let action = 0; // action -1 para abaixar, 1 para pular e 0 para andar
      dino.lastState = state;
      const prediction = dino.model.predict(tf.tensor2d([convertStateToVector(state)]));
      const predictionPromise = prediction.data();
      
      predictionPromise.then((result) => {
        console.log(result)       
        if ( (result > 0.40) && (result < 0.60) ) {
          action = 1;
          dino.lastJumpingState = state;
        } else if ((result <= 0.40)) {
          action = -1;
          dino.lastDuckingState = state;
        } else {
          dino.lastRunningState = state;
        }
        resolve(action);
      });
    } else {
      resolve(0);
    }
  });
}
/**
 * 
 * @param {object} dino 
 * handles the crash of a dino before restarting the game
 * 
 */
function handleCrash( dino ) {
  let input = null;
  let label = null;
  if (dino.jumping) {
    // input = convertStateToVector(dino.lastJumpingState);
    input = convertStateToVector(dino.lastState);
    label = [1];
  } else if (dino.ducking){ 
    // input = convertStateToVector(dino.lastDuckingState);
    input = convertStateToVector(dino.lastState);
    label = [-1];
  }else {
    // input = convertStateToVector(dino.lastRunningState);
    input = convertStateToVector(dino.lastState);
    label = [0];
  }
  dino.training.inputs.push(input);
  dino.training.labels.push(label);
}

/**
 * 
 * @param {object} state
 * returns an array 
 * converts state to a feature scaled array
 */
function convertStateToVector(state) {
  let retorno = [0, 0, 0, 0];
  if (state) {
    retorno = [
      state.obstacleX / CANVAS_WIDTH,
      state.obstacleHeight / CANVAS_HEIGHT,
      state.obstacleWidth / CANVAS_WIDTH,
      state.speed / 100
    ];
  }
  // console.log(retorno);
  // console.log(state);
  return retorno;
}
// call setup on loading content
document.addEventListener('DOMContentLoaded', setup);