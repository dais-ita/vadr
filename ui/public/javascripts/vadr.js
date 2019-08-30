let lastTs = null;
let activities = {};
let explanations = {};
let insights = {};
let foresights = {};

const DEBUG = false;
const POLL_DELAY = 500;
const LIST_INSIGHTS = '#list-insights';
const LIST_FORESIGHTS = '#list-foresights';
const EXP_SUMM = '#expSummary';
const EXP_SUMMTEXT = '#expSummaryText';
const EXP_VIDEO = '#expVideo';
const EXP_AUDIO = '#expAudio';
const EXP_INF = '#expInferences';
const EXP_WIDTH = '150px';
const SERVER = 'http://localhost:3000';
const URL_RESET = SERVER + '/reset?load_test_data=true';
const URL_ACTIVITIES = SERVER + '/activities';
const URL_EXPLANATIONS= SERVER + '/explanations';
const URL_EXP= SERVER + '/explain?activity_id=';
const PARM_NORANDOM = '&no_random=true';

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
                if (!insights[exp.activity.detection_id]) {
                    insights[exp.activity.detection_id] = exp;
                    updateInsightsList(exp);
                }
            }

            if (exp.foresight) {
                if (!foresights[exp.activity.detection_id]) {
                    foresights[exp.activity.detection_id] = exp;
                    updateForesightsList(exp);
                }
            }
        });
    }
}

function populateInsightsList(data) {
    let list = $(LIST_INSIGHTS);

    $.each(data.result.activities, function() {
        list.append($('<option />').val(this.detection_id).text(descriptionForInsight(this)));
    });
}

function updateInsightsList(exp) {
    $(LIST_INSIGHTS + ' option').each(function() {
        if (this.value == exp.activity.detection_id) {
            this.text = exp.insight.insight_summary + ' ' + timestampTextFor(exp.activity.detection_timestamp);
        }
    });
}

function updateForesightsList(exp) {
    let list = $(LIST_FORESIGHTS);

    list.append($('<option />').val(exp.foresight.foresight_detection).text(exp.foresight.foresight_summary));
}

function selectInsight() {
    let activityId = $(LIST_INSIGHTS).val();

    if (activityId) {
        showExplanation(explanations[activityId]);
    } else {
        hideExplanation();
    }
}

function selectForesight() {
    let activityId = $(LIST_FORESIGHTS).val();

    if (activityId) {
        showExplanation(explanations[activityId]);
    } else {
        hideExplanation();
    }
}

function showExplanation(exp) {
    showSummaryExplanation(exp);
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

function showSummaryExplanation(exp) {
    $(EXP_SUMM).removeClass('d-none');

    $(EXP_SUMMTEXT).html(descriptionForExplanation(exp));
}

function showVideoExplanation(exp) {
    let expVideo = $(EXP_VIDEO);

    if (exp.video) {
        if (exp.video.video_url) {
            let html = '';

            html += '<h6>Video:</h6>';
            html += '<img src="' + exp.video.video_url + '" width="' + EXP_WIDTH+ '"></img>';

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
            html += '<h6>Audio:</h6>';
            html += '<img src="' + exp.audio.audio_image_url + '" width="' + EXP_WIDTH + '"></img>';
        }

        if (exp.audio.audio_sound_url) {
            html += '<audio controls="controls" style="width:' + EXP_WIDTH + '"><source src="' + exp.audio.audio_sound_url + '" type="audio/ogg">Your browser does not support the audio element.</audio>';
        }

        if (exp.audio.audio_matched_words) {
            html += '<br/><h6>Words heard: ' + exp.audio.audio_matched_words + '</h6>';
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
            let fText = exp.insight.insight.replace(new RegExp('\n', 'g'), '<BR/>')
            html += '<li>' + fText + '</li>';
        }

        if (exp.foresight) {
            let fText = exp.foresight.foresight.replace(new RegExp('\n', 'g'), '<BR/>')
            html += '<li>' + fText + '</li>';
        }

        html += '</ol>';

        expInf.html(html);
        expInf.removeClass('d-none');
    } else {
        expInf.addClass('d-none');
    }
}

function buttonReset(norandom) {
    let resetUrl = URL_RESET;

    if (norandom) {
        resetUrl += PARM_NORANDOM;
    }

    hideExplanation();

    activities = {};
    explanations = {};
    insights = {};
    foresights = {};
    $(LIST_INSIGHTS).empty();
    $(LIST_FORESIGHTS).empty();

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

function buttonInsights() {
    console.log(insights);
}

function buttonForesights() {
    console.log(foresights);
}

function descriptionForInsight(activity) {
    return 'activity detected (' + activity.activity_id + ') ' + timestampTextFor(activity.detection_timestamp);
}

function descriptionForExplanation(exp) {
    let label = null;
    let insight = insights[exp.activity.detection_id];

    if (insight) {
        label = insight.insight.insight_summary;
    } else {
        label = exp.detected.name;
    }
    return label + ' ' + timestampTextFor(exp.activity.detection_timestamp);
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