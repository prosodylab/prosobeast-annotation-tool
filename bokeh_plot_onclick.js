var audioElement = document.getElementById("audio");
var i = cb_obj.indices;
console.log('clicked '+i)
if ($.isEmptyObject(i)) {
    r.active = null;
} else {
    var label = s.data['label'][i]
    console.log(label)
    r.active = ls.indexOf(label);
}
r.change.emit();
if (!$.isEmptyObject(i)) {
    $.ajax({
        type: "POST",
        url: document.URL + 'contour_clicked',
        data: {'index': i},
        success: function callback(response){
            console.log(response);
            if ($.isEmptyObject(i)) {
                audioElement.src = (document.URL
                    + 'static/ping.wav');
            } else {
                audioElement.src = (document.URL
                    + 'static/audio/'
                    + response + '.wav');
                var promise = audioElement.play();
                if (promise !== undefined) {
                    promise.then(_ => {
                        console.log('playing')
                    }).catch(error => {
                        console.log('playing error')
                        console.log(error)
                        console.log(error.message)
                    });
                }
            };
        },
        error: function (xhr, ajaxOptions, thrownError) {
            console.log(xhr.status);
            console.log(ajaxOptions);
            console.log(thrownError);
        },
    });
};
