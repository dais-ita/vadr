const express = require('express');
const router = express.Router();
const fs = require('fs-extra');
const path = require('path');
const config = require('../config.js');
const fn = require('../functions.js');

/* GET home page. */
router.get('/', function(req, res, next) {
  let event = fn.createEvent();
  let response = fn.createResponse(req, event);

  event.srcFolder = path.resolve(__dirname, '..' + config.testFolder);
  event.tgtFolder = path.resolve(__dirname, '..' + config.detectedFolder);
  event.deleteCount = 0;
  event.copyCount = 0;

  deleteExistingFiles(config, event, config.detectedFolder);
  deleteExistingFiles(config, event, config.audioFolder);
  deleteExistingFiles(config, event, config.videoFolder);
  deleteExistingFiles(config, event, config.insightFolder);
  deleteExistingFiles(config, event, config.foresightFolder);

  if (response.parameters.load_test_data == "true") {
    copyTestFiles(config, event, response);
  }

  response.result.status = 'success';
  response.result.filesCopied = event.copyCount;
  response.result.filesDeleted = event.deleteCount;

  res.json(response);
});

function deleteExistingFiles(config, event, folder) {
  let tgtFolder = path.resolve(__dirname, '..' + folder);
  let oldFiles = fs.readdirSync(tgtFolder);

  for (let i in oldFiles) {
    let filename = oldFiles[i];
    fs.unlinkSync(tgtFolder + config.file_separator + filename);
    ++event.deleteCount;
  }
}

function copyTestFiles(config, event, response) {
  let srcFiles = fs.readdirSync(event.srcFolder);

  for (let i in srcFiles) {
    let filename = srcFiles[i];
    let srcFn = event.srcFolder + config.file_separator + filename;
    let templatedActivity = fn.loadFromFile(srcFn);
    let convertedActivity = convertActivity(templatedActivity, config, event, response);

    let audioObj = null;
    let videoObj = null;
    let insightObj = null;
    let foresightObj = null;

    if (convertedActivity.video_url) {
      if (videoObj == null) {
        videoObj = {};
        videoObj.detection_id = convertedActivity.detection_id;
      }
      videoObj.video_url = convertedActivity.video_url;
      delete convertedActivity.video_url;
    }

    if (convertedActivity.audio_image_url) {
      if (audioObj == null) {
        audioObj = {};
        audioObj.detection_id = convertedActivity.detection_id;
      }
      audioObj.audio_image_url = convertedActivity.audio_image_url;
      delete convertedActivity.audio_image_url;
    }

    if (convertedActivity.audio_sound_url) {
      if (audioObj == null) {
        audioObj = {};
        audioObj.detection_id = convertedActivity.detection_id;
      }
      audioObj.audio_sound_url = convertedActivity.audio_sound_url;
      delete convertedActivity.audio_sound_url;
    }

    if (convertedActivity.audio_matched_words) {
      if (audioObj == null) {
        audioObj = {};
        audioObj.detection_id = convertedActivity.detection_id;
      }
      audioObj.audio_matched_words = convertedActivity.audio_matched_words;
      delete convertedActivity.audio_matched_words;
    }

    if (convertedActivity.insight) {
      if (insightObj == null) {
        insightObj = {};
        insightObj.detection_id = convertedActivity.detection_id;
      }
      insightObj.insight = convertedActivity.insight;
      insightObj.insight_summary = convertedActivity.insight_summary;
      delete convertedActivity.insight;
      delete convertedActivity.insight_summary;
    }

    if (convertedActivity.foresight) {
      if (foresightObj == null) {
        foresightObj = {};
        foresightObj.detection_id = convertedActivity.detection_id;
      }
      foresightObj.foresight = convertedActivity.foresight;
      foresightObj.foresight_summary = convertedActivity.foresight_summary;
      foresightObj.foresight_detection = convertedActivity.foresight_detection;
      delete convertedActivity.foresight;
      delete convertedActivity.foresight_summary;
      delete convertedActivity.foresight_detection;
    }

    if (videoObj != null) {
      writeFile(config, event, path.resolve(__dirname, '..' + config.videoFolder), videoObj);
    }

    if (audioObj != null) {
      writeFile(config, event, path.resolve(__dirname, '..' + config.audioFolder), audioObj);
    }

    if (insightObj != null) {
      writeFile(config, event, path.resolve(__dirname, '..' + config.insightFolder), insightObj);
    }

    if (foresightObj != null) {
      writeFile(config, event, path.resolve(__dirname, '..' + config.foresightFolder), foresightObj);
    }

    writeFile(config, event, event.tgtFolder, convertedActivity);
  }
}

function writeFile(config, event, tgtFolder, obj) {
  let tgtFn = tgtFolder + config.file_separator + obj.detection_id + '.json';

  fs.writeFileSync(tgtFn, JSON.stringify(obj, null, 2), config.encoding);
  ++event.copyCount;
}

function convertActivity(ta, config, event, response) {
  let result = {};

  // {id}       - replaced with value of detection_id property [A]
  // {now}      - replaced with current timestamp [B]
  // +/- nnnn   - use this number as an offset to increment the timestamp [C]
  // {rand:x-y} - replaced with random integer between x and y inclusive [D]
  // |x         - use this number instead if no_random=true specified as url parameter [E]
  // {root}     - replaced with rootUrl [F]

  for (let prop in ta) {
    if (Object.prototype.hasOwnProperty.call(ta, prop)) {
      let finalValue = ta[prop];

      // [A]
      if (finalValue.includes('{id}')) {
        finalValue = finalValue.replace('{id}', ta.detection_id);
      }

      // [B]
      if (finalValue == '{now}') {
        finalValue = event.ts;
      } else {

        // [C]
        if (finalValue.includes('{now}')) {
          let remText = finalValue.replace('{now}', '');
          let remNum = parseInt(remText, 10);

          finalValue = event.ts + remNum;
        } else {
          // [F]
          if (finalValue.includes('{root}')) {
            finalValue = finalValue.replace('{root}', config.root_url);
          }

          // [D], [E]
          if (finalValue.includes('{rand:')) {
            let randParms = finalValue.replace('{rand:', '');
            let parms = randParms.split('}');

            let randBounds = parms[0].split('-');
            let randAlternative = null;

            if (parms.length > 1) {
              randAlternative = parms[1].replace('|', '');
            }

            if (response.parameters.no_random == "true") {
              // [E]
              finalValue = parseInt(randAlternative);
            } else {
              // [D]
              let randLower = parseInt(randBounds[0]);
              let randUpper = parseInt(randBounds[1]);
              let randRange = randUpper - randLower;
              let rawRand = Math.random();
              finalValue = Math.round(randLower + (rawRand * randRange));
            }
          }
        }
      }

      result[prop] = finalValue;
    }
  }

  return result;
}

module.exports = router;
