// input is
// s=source,
// r=locations_group,
// location_labels=location_labels,
// xs=xs,
// ys=ys,
// scales=scales,
// x_slider=x_slider,
// y_slider=y_slider,
var label = location_labels[r.active];
console.log(label);
var len = s.data['x'].length;
for (var i = 0; i < len; i++) {
    var x = xs.data[label][i];
    var y = ys.data[label][i];
    s.data['x'][i] = x;
    s.data['y'][i] = y;
};
// set the scale on the sliders to the default
x_slider.value = scales.data[label][0]
y_slider.value = scales.data[label][1]
s.change.emit();
