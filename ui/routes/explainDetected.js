const express = require('express');
const router = express.Router();
const config = require('../config.js');
const fn = require('../functions.js');
const ucf101 = require('../classes/UCF101.json');

/* GET home page. */
router.get('/', function(req, res, next) {
  let event = fn.createEvent();
  let response = fn.createResponse(req, event);
  let fullFn = '.' + config.detectedFolder + response.parameters.activity_id + config.suffix;

  let thisActivity = fn.loadFromFile(fullFn);

  response.result = fn.getUcf101ClassFor(thisActivity.activity_id, ucf101);

  res.json(response);
});

module.exports = router;
