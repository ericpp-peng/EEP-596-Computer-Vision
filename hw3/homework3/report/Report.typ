// =======================================================
// EE P 596  Computer Vision: Classical and Deep Methods task_outputs Report Template (Typst)
// Author: Po Peng (Eric)
// =======================================================

// --- Metadata ---
#let course = "EEP 596A 
Computer Vision: Classical and Deep Methods"
#let quarter = "2025 Fall"
#let hw_title = "Homework 3 Report"
#let name = "Po Peng"
#let netid = "ericpp"

// --- Global Styles ---
#set page(width: 8.5in, height: 11in, margin: 1in)
#set text(size: 11pt)
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

= Task 1 – Basic tensor arithmetic (saturation)
#figure(
  image("../task_outputs/Task1c_saturation.png", width: 4in),
  caption: [Task 1 (1a/1b/1c): Basic tensor arithmetic. Shown: Task 1c — saturation arithmetic producing uint8 result after adding 100 to each channel]
)

= Task 2 – Add Gaussian noise
#figure(
  image("../task_outputs/Task2_add_noise.png", width: 4in),
  caption: [Task 2: Image with additive Gaussian noise (mean=0, sigma=100) — displayed as float32 normalized]
)

= Task 3 – Image normalization
#align(center)[
  #figure(image("../task_outputs/Task3b_Imagenet_norm.png", width: 4in), caption: [Task 3b: Normalized using ImageNet means/stds])
]

 = Task 4 – Dimension rearranging
#align(center)[
  #figure(
    image("../task_outputs/Task4_rearrange.png", width: 4in),
    caption: [Task 4: Tensor rearranged to NxCxHxW (N=1,C=3,H,W shown visually)]
  )
]

= Task 5 – Stride convolution with Scharr_x filter
#align(center)[
  #figure(
    image("../task_outputs/Task5_stride.png", width: 4in),
    caption: [Task 5: Grayscale image convolved with Scharr_x and stride=2 result]
  )
]