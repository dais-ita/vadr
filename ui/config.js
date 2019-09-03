let config = {};

config.file_separator = '/';
config.encoding = 'utf8'

config.port = 3000;
config.hostname = 'localhost';
config.root_url = 'http://' + config.hostname + ':' + config.port;
config.suffix = '.json';

config.testFolder = '/events/activities/test_activities/';
config.detectedFolder = '/events/activities/detected_activities/';
config.audioFolder = '/events/explanations/audio/';
config.videoFolder = '/events/explanations/video/';
config.insightFolder = '/events/explanations/insight/';
config.foresightFolder = '/events/explanations/foresight/';

config.sampleStream = 'https://www.youtube.com/watch?v=FOajbVDTRlU';

//Sample audio from http://shtooka.net/

module.exports = config;