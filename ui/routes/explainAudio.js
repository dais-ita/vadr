const express = require('express');
const router = express.Router();
const config = require('../config.js');
const fn = require('../functions.js');

/* GET home page. */
router.get('/', function(req, res, next) {
  let event = fn.createEvent();
  let response = fn.createResponse(req, event);
  let fullFn = '.' + config.audioFolder + response.parameters.activity_id + config.suffix;

  response.result = fn.loadFromFile(fullFn);

  res.json(response);
});

module.exports = router;
