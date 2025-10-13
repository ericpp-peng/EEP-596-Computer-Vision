// =======================================================
// EE P 596  Computer Vision: Classical and Deep Methods task_outputs Report Template (Typst)
// Author: Po Peng (Eric)
// =======================================================

// --- Metadata ---
#let course = "EEP 596A 
Computer Vision: Classical and Deep Methods"
#let quarter = "2025 Fall"
#let hw_title = "Homework 2 Report"
#let name = "Po Peng"
#let netid = "ericpp"

// --- Global Styles ---
#set page(width: 8.5in, height: 11in, margin: 1in)
#set text(font: "sans", size: 11pt)
#show heading: set text(14pt, weight: "bold")
#show figure.caption: set text(10pt, gray)
#show "```": block.with(fill: luma(98%), inset: 6pt, radius: 4pt, stroke: luma(90%))

#let today = datetime.today()


// --- Cover Page ---
#align(center)[
  #v(2em)
  #set text(size: 20pt, weight: "bold")
  #(course)
  #v(-0.5em)
  #set text(size: 14pt)
  #(quarter)
  #v(2em)
  #set text(size: 20pt, weight: "bold")
  #(hw_title)
  #v(2em)
  #set text(size: 16pt)
  *Name:* #name \
  *NetID:* #netid \
  *Date:* #today.display("[month repr:long] [day], [year]") \
  #v(6em)
  #set text(size: 11pt)
]

// --- Table of Contents ---
#v(0.5em)
Contents
#v(0.2em)
#outline(title: [], depth: 2)

// --- Tasks ---
#v(1em)
#v(1em)
#v(1em)
#v(1em)
#v(1em)
#v(1em)
#v(1em)
#v(1em)

= Task 1 – Load and Analyze Image
#figure(
  image("../task_outputs/Task1_floodfilled.jpg", width: 4in),
  caption: [Floodfilled ant face (seed-based stack floodfill result)]
)

= Task 2 – Gaussian Smoothing (repeated)
#figure(image("../task_outputs/Task2_blur_0.jpg", width: 1.6in), caption: [Gaussian smoothing — level 0 (original)])
#figure(image("../task_outputs/Task2_blur_1.jpg", width: 1.6in), caption: [Gaussian smoothing — level 1])
#figure(image("../task_outputs/Task2_blur_2.jpg", width: 1.6in), caption: [Gaussian smoothing — level 2])
#figure(image("../task_outputs/Task2_blur_3.jpg", width: 1.6in), caption: [Gaussian smoothing — level 3])
#figure(image("../task_outputs/Task2_blur_4.jpg", width: 1.6in), caption: [Gaussian smoothing — level 4])

= Task 3 – Vertical Derivative (Gaussian × derivative)
#figure(image("../task_outputs/Task3_vertical_derivative_0.jpg", width: 1.6in), caption: [Vertical derivative — level 0])
#figure(image("../task_outputs/Task3_vertical_derivative_1.jpg", width: 1.6in), caption: [Vertical derivative — level 1])
#figure(image("../task_outputs/Task3_vertical_derivative_2.jpg", width: 1.6in), caption: [Vertical derivative — level 2])
#figure(image("../task_outputs/Task3_vertical_derivative_3.jpg", width: 1.6in), caption: [Vertical derivative — level 3])
#figure(image("../task_outputs/Task3_vertical_derivative_4.jpg", width: 1.6in), caption: [Vertical derivative — level 4])

= Task 4 – Horizontal Derivative (transpose of previous)
#figure(image("../task_outputs/Task4_horizontal_derivative_0.jpg", width: 1.6in), caption: [Horizontal derivative — level 0])
#figure(image("../task_outputs/Task4_horizontal_derivative_1.jpg", width: 1.6in), caption: [Horizontal derivative — level 1])
#figure(image("../task_outputs/Task4_horizontal_derivative_2.jpg", width: 1.6in), caption: [Horizontal derivative — level 2])
#figure(image("../task_outputs/Task4_horizontal_derivative_3.jpg", width: 1.6in), caption: [Horizontal derivative — level 3])
#figure(image("../task_outputs/Task4_horizontal_derivative_4.jpg", width: 1.6in), caption: [Horizontal derivative — level 4])

= Task 5 – Gradient Magnitude (Manhattan norm)
#figure(image("../task_outputs/Task5_gradmag_0.jpg", width: 1.6in), caption: [Gradient magnitude — level 0])
#figure(image("../task_outputs/Task5_gradmag_1.jpg", width: 1.6in), caption: [Gradient magnitude — level 1])
#figure(image("../task_outputs/Task5_gradmag_2.jpg", width: 1.6in), caption: [Gradient magnitude — level 2])
#figure(image("../task_outputs/Task5_gradmag_3.jpg", width: 1.6in), caption: [Gradient magnitude — level 3])
#figure(image("../task_outputs/Task5_gradmag_4.jpg", width: 1.6in), caption: [Gradient magnitude — level 4])

= Task 6 – Gaussian Derivative (scipy.signal.convolve2d)
#figure(image("../task_outputs/Task6_scipy_vertical_derivative_0.jpg", width: 1.6in), caption: [SciPy vertical derivative — level 0])
#figure(image("../task_outputs/Task6_scipy_vertical_derivative_1.jpg", width: 1.6in), caption: [SciPy vertical derivative — level 1])
#figure(image("../task_outputs/Task6_scipy_vertical_derivative_2.jpg", width: 1.6in), caption: [SciPy vertical derivative — level 2])
#figure(image("../task_outputs/Task6_scipy_vertical_derivative_3.jpg", width: 1.6in), caption: [SciPy vertical derivative — level 3])
#figure(image("../task_outputs/Task6_scipy_vertical_derivative_4.jpg", width: 1.6in), caption: [SciPy vertical derivative — level 4])

= Task 7 – Repeated Box Filtering (approximate Gaussian)
#figure(
  image("../task_outputs/Task7_box_powers.png", width: 4in),
  caption: [Repeated box filter convolutions (power of box filter) illustrating approach to Gaussian]
)