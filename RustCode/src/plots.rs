use gnuplot::{Figure, Caption, Color};
use plotlib;
use plotlib::page::Page;
//use plotlib::repr::ContinuousRepresentation;
//use plotlib::repr::Plot;
use plotlib::view::ContinuousView;


#[allow(dead_code)]
pub fn plot_lines(x:&Vec<f64>, y:&Vec<f64>){
    let mut fg = Figure::new();
    let z:Vec<usize> = (0..x.len()).collect();
    let axes = fg.axes2d();
    axes.lines(&z, x, &[Caption("True states"), Color("green")]);
    axes.lines(&z, y, &[Caption("Est. states"), Color("blue")]);
    fg.show().expect(" plotting error");
}

#[allow(dead_code)]
pub fn plot_err_line(x:&Vec<f64>){
    let mut fg = Figure::new();
    let z:Vec<usize> = (0..x.len()).collect();
    let axes = fg.axes2d();
    axes.lines(&z, x, &[Caption("Err in x"), Color("green")]);
    fg.show().expect(" plotting error");
}

#[allow(dead_code)]
pub fn plot_a_line(x:&Vec<f64>, y:&Vec<f64>){
    let mut fg = Figure::new();
    let axes = fg.axes2d();
    axes.lines(x, y, &[Caption("A line"), Color("green")]);
    fg.show().expect(" plotting error");
}

#[allow(dead_code)]
pub fn plot_hist(x:&Vec<f64>){

    //let z:Vec<f64> = vec![10.0;10];//(0..x.len()).collect();
    let h = plotlib::repr::Histogram::from_slice(x, plotlib::repr::HistogramBins::Count(100));
    //let h = plotlib::repr::Histogram::from_slice(&z, plotlib::repr::HistogramBins::Count(30));
    let v = ContinuousView::new()
        .add(h)
        .x_range(-5.0, 5.0)
        .y_range(-5.0, 5.0)
        .x_label("Some varying variable")
        .y_label("The response of something");

    // A page with a single view is then saved to an SVG file
    Page::single(&v).save("pg.svg").unwrap();
}
