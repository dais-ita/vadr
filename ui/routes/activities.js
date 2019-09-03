const express = require('express');
const router = express.Router();
const fs = require('fs-extra');
const path = require('path');
const config = require('../config.js');
const fn = require('../functions.js');

/* GET home page. */
router.get('/', function(req, res, next) {
  const detectedFolder = path.resolve(__dirname, '..' + config.detectedFolder);
  let event = fn.createEvent();
  let response = fn.createResponse(req, event);

  if (response.parameters.since) {
    event.sinceTs = response.parameters.since;
  }

  response.result.activities = [];

  processRawActivities(detectedFolder, response.result.activities, config, event);

  res.json(response);
});

function processRawActivities(folder, activities, config, event) {
  let files = fs.readdirSync(folder);

  for (let i in files) {
    let fullFn = folder + config.file_separator + files[i];
    let thisActivity = fn.loadFromFile(fullFn);
    let inRange = true;

    //If a since parameter was specified, is this activity in range (i.e. after the since timestamp)
    if (event.sinceTs) {
      inRange = thisActivity.detection_timestamp >= event.sinceTs;
    }

    //If this activity is in range so far, is it still in range (i.e. timestamp before current timestamp)
    if (inRange) {
      inRange = thisActivity.detection_timestamp <= event.ts;
    }

    //If this activity is still in range then add it to the list of matched activities
    if (inRange) {
      activities.push(thisActivity);
    }
  }
}

module.exports = router;
