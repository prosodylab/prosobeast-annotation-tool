// s=source,
// r=locations_group,
// location_labels=location_labels,
// xs=xs,
// ys=ys,
// scales=scales,
// x_slider=x_slider,
// y_slider=y_slider,
var label = location_labels[r.active];
var x_scale = x_slider.value;
var y_scale = scales.data[label][1];
let source_label = 'location';
if (label != 'user') {
    source_label = `${source_label}_${label}`
}
var len = s.data['x'].length;
// console.log(s.data);
console.log(source_label);
// console.log(s.data[source_label]);
for (var i = 0; i < len; i++) {
    // x and y from location_dict
    var location = JSON.parse(s.data[source_label][i]);
    var x_loc = location[0]
    var y_loc = location[1]

    var f0 = s.data['f0'][i];

    var contour_len = s.data['length'][i];
    var x = linspace(-1, 1, contour_len)
    x = x.map(o => o * x_scale);
    x = x.map(o => o + x_loc);
    var y = f0.map(o => o * y_scale);
    y = y.map(o => o + y_loc);

    s.data['x'][i] = x
    s.data['y'][i] = y
    xs.data[label][i] = x
    ys.data[label][i] = y
};

scales.data[label][0] = x_scale
scales.data[label][1] = y_scale

s.change.emit();
scales.change.emit();
xs.change.emit();
ys.change.emit();

