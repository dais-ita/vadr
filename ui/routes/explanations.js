const express = require('express');
const router = express.Router();
const fs = require('fs-extra');
const path = require('path');
const config = require('../config.js');
const fn = require('../functions.js');
const ucf101 = require('../classes/UCF101.json');

/* GET home page. */
router.get('/', function(req, res, next) {
  let event = fn.createEvent();
  let response = fn.createResponse(req, event);

  if (response.parameters.since) {
    event.sinceTs = response.parameters.since;
  }

  response.result.explanations = [];

  processRawExplanations(response.result.explanations, config, event);

  res.json(response);
});

function processRawExplanations(explanations, config, event) {
  const detectedFolder = path.resolve(__dirname, '..' + config.detectedFolder);
  const audioFolder = path.resolve(__dirname, '..' + config.audioFolder);
  const videoFolder = path.resolve(__dirname, '..' + config.videoFolder);
  const insightFolder = path.resolve(__dirname, '..' + config.insightFolder);
  const foresightFolder = path.resolve(__dirname, '..' + config.foresightFolder);
  let files = fs.readdirSync(detectedFolder);

  for (let i in files) {
    let fullFn = detectedFolder + config.file_separator + files[i];
    let audioFn = audioFolder + config.file_separator + files[i];
    let videoFn = videoFolder + config.file_separator + files[i];
    let insightFn = insightFolder + config.file_separator + files[i];
    let foresightFn = foresightFolder + config.file_separator + files[i];
    let thisExp = {};
    thisExp.activity = fn.loadFromFile(fullFn);
    thisExp.detected = fn.getUcf101ClassFor(thisExp.activity.activity_id, ucf101);

    if (fs.existsSync(audioFn)) {
      thisExp.audio = fn.loadFromFile(audioFn);
    }

    if (fs.existsSync(videoFn)) {
      thisExp.video = fn.loadFromFile(videoFn);
    }

    if (fs.existsSync(insightFn)) {
      thisExp.insight = fn.loadFromFile(insightFn);
    }

    if (fs.existsSync(foresightFn)) {
      thisExp.foresight = fn.loadFromFile(foresightFn);
    }

    let inRange = true;

    //If a since parameter was specified, is this activity in range (i.e. after the since timestamp)
    if (event.sinceTs) {
      inRange = thisExp.activity.detection_timestamp >= event.sinceTs;
    }

    //If this activity is in range so far, is it still in range (i.e. timestamp before current timestamp)
    if (inRange) {
      inRange = thisExp.activity.detection_timestamp <= event.ts;
    }

    //If this activity is still in range then add it to the list of matched activities
    if (inRange) {
      explanations.push(thisExp);
    }
  }
}

module.exports = router;
