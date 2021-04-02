var i = s.selected.indices;
if (i.length == 0) {
    r.active = null;
    new_label = null;
} else {
    var new_label = ls[r.active];
    var new_color = cs[r.active];
    // new_color = cs[lsp.indexOf(new_label)];
    s.data['color'][i] = new_color;
    s.data['label'][i] = new_label;
};
s.change.emit();
$.ajax({
    type: "POST",
    url: document.URL + 'form_clicked',
    //data: JSON.stringify({'index': i, 'new_label': new_label}),
    data: {'index': i, 'new_label': new_label},
    success: function callback(response){
        console.log(response);
        if (response.startsWith("Change!")){
            var button = $("#download_data_button");
            blue_to_gray(button)
            var span = $("#span_download_data");
            span.text("Changes not saved!");
        }
    },
    error: function (xhr, ajaxOptions, thrownError) {
        console.log(xhr.status);
        console.log(thrownError);
    },
});
