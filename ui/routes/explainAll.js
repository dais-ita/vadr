const express = require('express');
const router = express.Router();
const config = require('../config.js');
const fn = require('../functions.js');

/* GET home page. */
router.get('/', function(req, res, next) {
  let event = fn.createEvent();
  let response = fn.createResponse(req, event);
  let urlParms = '?activity_id=' + response.parameters.activity_id;

  event.rawResponse = [];
  event.numResponses = 0;
  event.maxResponses = 5;

  fn.makeHttpRequest(config,1, 'detected', '/explain/detected' + urlParms, event, res, response);
  fn.makeHttpRequest(config,2, 'audio', '/explain/audio' + urlParms, event, res, response);
  fn.makeHttpRequest(config,3, 'video', '/explain/video' + urlParms, event, res, response);
  fn.makeHttpRequest(config,4, 'insight', '/explain/insight' + urlParms, event, res, response);
  fn.makeHttpRequest(config,5, 'foresight', '/explain/foresight' + urlParms, event, res, response);
});

module.exports = router;
