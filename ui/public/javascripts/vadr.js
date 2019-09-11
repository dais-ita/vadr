let lastTs = null;
let activities = {};
let explanations = {};
let identified = {};
let predicted = {};

const DEBUG = false;
const POLL_DELAY = 500;
const LIST_DETECTED = '#list-detected';
const LIST_IDENTIFIED = '#list-identified';
const LIST_PREDICTED = '#list-predicted';
const EXP_SUMM = '#expSummary';
const EXP_SUMMTEXT = '#expSummaryText';
const EXP_VIDEO = '#expVideo';
const EXP_AUDIO = '#expAudio';
const EXP_INF = '#expInferences';
const EXP_WIDTH = '150px';
const SERVER = 'http://localhost:3000';
const URL_RESET = SERVER + '/reset';
const URL_ACTIVITIES = SERVER + '/activities';
const URL_EXPLANATIONS= SERVER + '/explanations';
//const URL_EXP= SERVER + '/explain?activity_id=';
const PARM_NORANDOM = '&no_random=true';
const PARM_NODATA = '?load_test_data=';

const MODE_DET = 0;
const MODE_IDENT = 1;
const MODE_PRED = 2;

//$('#band_type_choices option[name="acoustic"]').text('Wedding Ceremony');

function initialise() {
    pollForItems();
}

function pollForItems() {
    let activityUrl = URL_ACTIVITIES;
    let explanationUrl = URL_EXPLANATIONS;

    if (lastTs) {
        activityUrl += '?since=' + lastTs;
        //Don't use the sine parameter for explanations as they can change later
    }

    $.get(activityUrl, function(data) {
        populateActivities(data.result.activities);
        populateInsightsList(data);
        setTimeout(pollForItems, POLL_DELAY);

        lastTs = data.timestamp;
    });

    $.get(explanationUrl, function(data) {
        populateExplanations(data.result.explanations);
    });
}

function populateActivities(actList) {
    if (actList.length > 0) {
        $.each(actList, function( i, activity ) {
            activities[activity.detection_id] = activity;
        });
    }
}

function populateExplanations(expList) {
    if (expList.length > 0) {
        $.each(expList, function( i, exp ) {
            explanations[exp.activity.detection_id] = exp;

            if (exp.insight) {
                if (!identified[exp.activity.detection_id]) {
                    identified[exp.activity.detection_id] = exp;
                    updateInsightsList(exp);
                }
            }

            if (exp.foresight) {
                if (!predicted[exp.activity.detection_id]) {
                    predicted[exp.activity.detection_id] = exp;
                    updateForesightsList(exp);
                }
            }
        });
    }
}

function populateInsightsList(data) {
    let list = $(LIST_DETECTED);

    $.each(data.result.activities, function() {
        list.append($('<option />').val(this.detection_id).text(descriptionForInsight(this)));
    });
}

function updateInsightsList(exp) {
    let list = $(LIST_IDENTIFIED);

    list.append($('<option />').val(exp.insight.detection_id).text(exp.insight.insight_summary + ' ' + timestampTextFor(exp.activity.detection_timestamp)));
}

function updateForesightsList(exp) {
    let list = $(LIST_PREDICTED);

    list.append($('<option />').val(exp.foresight.foresight_detection).text(exp.foresight.foresight_summary + ' ' + timestampTextFor(exp.activity.detection_timestamp)));
}

function selectDetected() {
    let activityId = $(LIST_DETECTED).val();

    if (activityId) {
        showExplanation(explanations[activityId], MODE_DET);
    } else {
        hideExplanation();
    }

    deselectIdentified();
    deselectPredicted();
}

function selectIdentified() {
    let activityId = $(LIST_IDENTIFIED).val();

    if (activityId) {
        showExplanation(explanations[activityId], MODE_IDENT);
    } else {
        hideExplanation();
    }

    deselectDetected();
    deselectPredicted();
}

function selectPredicted() {
    let activityId = $(LIST_PREDICTED).val();

    if (activityId) {
        showExplanation(explanations[activityId], MODE_PRED);
    } else {
        hideExplanation();
    }

    deselectDetected();
    deselectIdentified();
}

function deselectDetected() {
    $(LIST_DETECTED).find('option').prop('selected', false);
}

function deselectIdentified() {
    $(LIST_IDENTIFIED).find('option').prop('selected', false);
}

function deselectPredicted() {
    $(LIST_PREDICTED).find('option').prop('selected', false);
}

function showExplanation(exp, mode) {
    showSummaryExplanation(exp, mode);
    showVideoExplanation(exp);
    showAudioExplanation(exp);
    showInferenceExplanation(exp);
}

function hideExplanation() {
    $(EXP_SUMM).addClass('d-none');
    $(EXP_VIDEO).addClass('d-none');
    $(EXP_AUDIO).addClass('d-none');
    $(EXP_INF).addClass('d-none');
}

function showSummaryExplanation(exp, mode) {
    $(EXP_SUMM).removeClass('d-none');

    $(EXP_SUMMTEXT).html(descriptionForExplanation(exp, mode));
}

function showVideoExplanation(exp) {
    let expVideo = $(EXP_VIDEO);

    if (exp.video) {
        if (exp.video.video_url) {
            let html = '';

//            html += '<img src="' + exp.video.video_url + '" width="' + EXP_WIDTH+ '"></img>';
//            html += '<iframe class="embed-responsive-item" src="' + exp.video.video_url + '"></iframe>';
            html += '<video class="embed-responsive-item" controls autoplay loop>';
            html += '  <source src="' + exp.video.video_url + '" type="video/mp4">';
            html += '   Your browser does not support the video tag.';
            html += '</video>';

            expVideo.html(html);
            expVideo.removeClass('d-none');
        }
    } else {
        expVideo.addClass('d-none');
    }
}

function showAudioExplanation(exp) {
    let expAudio = $(EXP_AUDIO);

    if (exp.audio) {
        let html = '';

        if (exp.audio.audio_image_url) {
//            html += '<h6>Audio:</h6>';
//            html += '<img src="' + exp.audio.audio_image_url + '" width="' + EXP_WIDTH + '"></img>';
            html += '<img src="' + exp.audio.audio_image_url + '" width="100%"></img>';
        }

        expAudio.html(html);
        expAudio.removeClass('d-none');
    } else {
        expAudio.addClass('d-none');
    }
}

function showInferenceExplanation(exp) {
    let expInf = $(EXP_INF);

    if (exp.insight || exp.foresight) {
        let html = '';

        html += '<h6>Inferences:</h6>';
        html += '<ol>';

        if (exp.insight) {
            let fText = exp.insight.insight.replace(new RegExp('\n', 'g'), '<BR/>');
            html += '<li>' + fText + '</li>';
        }

        if (exp.foresight) {
            let fText = exp.foresight.foresight.replace(new RegExp('\n', 'g'), '<BR/>');
            html += '<li>' + fText + '</li>';
        }

        html += '</ol>';

        if (exp.audio.audio_sound_url) {
            html += '<br/><br/><br/>';
            html += '<audio controls="controls" style="width:' + EXP_WIDTH + '"><source src="' + exp.audio.audio_sound_url + '" type="audio/ogg">Your browser does not support the audio element.</audio>';
        }

        if (exp.audio.audio_matched_words) {
            html += '<br/><h6>Words heard: ' + exp.audio.audio_matched_words + '</h6>';
        }

        expInf.html(html);
        expInf.removeClass('d-none');
    } else {
        expInf.addClass('d-none');
    }
}

function buttonReset(norandom, loaddata) {
    let resetUrl = URL_RESET;

    if (loaddata) {
        resetUrl += PARM_NODATA + 'true';
    } else {
        resetUrl += PARM_NODATA + 'false';
    }

    if (norandom) {
        resetUrl += PARM_NORANDOM;
    }

    console.log(resetUrl);

    hideExplanation();

    activities = {};
    explanations = {};
    identified = {};
    predicted = {};
    $(LIST_DETECTED).empty();
    $(LIST_IDENTIFIED).empty();
    $(LIST_PREDICTED).empty();

    $.get(resetUrl, function(data) {
        debug(data);
    });
}

function buttonActivities() {
    console.log(activities);
}

function buttonExplanations() {
    console.log(explanations);
}

function buttonIdentified() {
    console.log(identified);
}

function buttonPredicted() {
    console.log(predicted);
}

function descriptionForInsight(activity) {
    return 'UCF101 class ' + activity.activity_id + ' ' + timestampTextFor(activity.detection_timestamp);
}

function descriptionForExplanation(exp, mode) {
    let label = 'Detected activity: ' + descriptionForDetected(exp);

    if (mode == MODE_IDENT) {
        let insight = identified[exp.activity.detection_id];

        if (insight) {
            label = 'Identified activity: ' + '<b>' + insight.insight.insight_summary + '</b> from detected activity ' + descriptionForDetected(exp);
        }
    }

    return label + ' ' + timestampTextFor(exp.activity.detection_timestamp);
}

function descriptionForDetected(exp) {
    return '<b>UCF101 class ' + exp.activity.activity_id + '</b>' + '<b> (' + exp.detected.name + ')</b>'
}

function timestampTextFor(ts) {
    let t = new Date(ts);

    return '[' + padNumTwo(t.getHours()) + ':' + padNumTwo(t.getMinutes()) + '.' + padNumTwo(t.getSeconds()) + ']';
}

function padNumTwo(num) {
    let result = null;

    if (num < 10) {
        result = '0' + num;
    } else {
        result = num.toString();
    }

    return result;
}

function debug(msg) {
    if (DEBUG) {
        console.log(msg);
    }
}