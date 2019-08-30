const http = require('http');

let fn = {};

fn.createEvent = function() {
    let result = {};

    result.ts = Date.now();

    return result;
};

fn.createResponse = function(req, event) {
    let obj = {};

    obj.timestamp = event.ts;
    obj.parameters = req.query;
    obj.result = {};

    return obj;
};

fn.loadFromFile = function(fn) {
    delete require.cache[require.resolve(fn)];  //Ensure the cache is cleared to the file is loaded fresh each time

    return require(fn);
};

fn.makeHttpRequest = function(config, i, label, url, event, mainRes, response) {
    const options = {
        hostname: config.hostname,
        port: config.port,
        path: url,
        method: 'GET'
    };

    const req = http.request(options, res => {
        res.on('data', d => {
            event.rawResponse[i] = d;
        });

        res.on('end', d => {
            let objRes = null;

            try {
                objRes = JSON.parse(event.rawResponse[i].toString());
            } catch(e) {
                objRes = null;
            }

            if (objRes && objRes.result) {
                response.result[label] = objRes.result;
            }

            ++event.numResponses;

            if (event.numResponses == event.maxResponses) {
                mainRes.json(response);
            }
        });
    });

    req.on('error', error => {
        console.error(error)
    });

    req.end();
};

fn.getUcf101ClassFor = function(tgtId, list) {
    for (let i = 0; i < list.length; i++) {
        let item = list[i];

        if (item.id == tgtId) {
            return item;
        }
    }
};

module.exports = fn;